import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)  # ← これを一番最初に！

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch

from eval_utils import load_model

from cdvae.pl_data.dataset import TensorCrystDataset
from torch_geometric.data import Batch

import torch.nn as nn

def _raw_property_preds(model, z):
    """
    MEGNet 用に ModuleList を束ねて (N,4) テンソルを返すヘルパ
    """
    if isinstance(model.fc_property, nn.ModuleList):
        return torch.cat([branch(z) for branch in model.fc_property], dim=1)
    else:                       # supercon など (N,1)
        return model.fc_property(z)

def build_data_object(model, crystal_dict):
    dataset = TensorCrystDataset(
        [crystal_dict],
        niggli=model.hparams.data.niggli,
        primitive=model.hparams.data.primitive,
        graph_method=model.hparams.data.graph_method,
        lattice_scale_method=model.hparams.data.lattice_scale_method,
        preprocess_workers=1,
    )
    dataset.scaler = model.scaler
    dataset.lattice_scaler = model.lattice_scaler
    return dataset[0]


def reconstructon(loader, model, ld_kwargs, num_evals,
                  force_num_atoms=False, force_atom_types=False, down_sample_traj_step=1):
    """
    reconstruct the crystals in <loader>.
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    input_data_list = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f'batch {idx} in {len(loader)}')
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        # only sample one z, multiple evals for stoichaticity in langevin dynamics
        _, _, z = model.encode(batch)

        for eval_idx in range(num_evals):
            gt_num_atoms = batch.num_atoms if force_num_atoms else None
            gt_atom_types = batch.atom_types if force_atom_types else None
            outputs = model.langevin_dynamics(
                z, ld_kwargs, gt_num_atoms, gt_atom_types)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lengths.append(outputs['lengths'].detach().cpu())
            batch_angles.append(outputs['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    outputs['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    outputs['all_atom_types'][::down_sample_traj_step].detach().cpu())
        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))
        # Save the ground truth structure
        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords, num_atoms, atom_types, lengths, angles,
        all_frac_coords_stack, all_atom_types_stack, input_data_batch)


def generation(model, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for z_idx in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size, model.hparams.hidden_dim,
                        device=model.device)

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(z, ld_kwargs)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)


######################## 変更後のイメージ ########################
def _sample_batch(model, ld_kwargs, z_batch, max_chunk=32):
    """
    z_batch: [B, latent_dim]
    戻り値: dict 各キーごとに B 個連結
    """
    try:
        out = model.langevin_dynamics(z_batch, ld_kwargs)
        return {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in out.items()}

    except RuntimeError as e:
        # triplet 0 件 or block_inc エラーが出たら再帰的に分割
        err = str(e)
        need_split = any(s in err for s in
                         ["block_inc contains", "input.numel() == 0"])
        if not need_split or z_batch.shape[0] == 1:
            raise          # 1 個でも落ちるなら諦める

        # ---- サブバッチに分割して再試行 ----
        mid = z_batch.shape[0] // 2
        left  = _sample_batch(model, ld_kwargs, z_batch[:mid])
        right = _sample_batch(model, ld_kwargs, z_batch[mid:])
        merged = {}
        for k in left:
            if torch.is_tensor(left[k]):
                merged[k] = torch.cat([left[k], right[k]], dim=0)
            else:
                merged[k] = left[k] + right[k]
        return merged
#################################################################

def optimization(model, ld_kwargs, data_loader,
                 num_starting_points=100, num_gradient_steps=5000,
                 lr=1e-3, num_saved_crys=10,
                 megnet_loss_mode=False,
                 coef_e_form=1.0 ,coef_100more=1.0, coef_tolerance=1.0):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
        z.requires_grad = True
    else:
        z = torch.randn(num_starting_points, model.hparams.hidden_dim,
                        device=model.device)
        z.requires_grad = True

    # 🔽 ここでタスクに応じて最大化／最小化を切り替え
    data_root = model.hparams.data.root_path if hasattr(model.hparams.data, "root_path") else ""
    maximize = "supercon" in data_root.lower()
    sign = -1.0 if maximize else 1.0

    # 🔽 ここで確認用に print
    print(f"[CDVAE OPTIMIZATION] Detected task: {data_root}")
    print(f"[CDVAE OPTIMIZATION] Optimization direction: {'maximize' if maximize else 'minimize'} (sign = {sign}) with learning rate: {lr}")
    print(f"[CDVAE OPTIMIZATION] num_starting_points:{num_starting_points}")
    target_bg = getattr(ld_kwargs, 'target_bg', -1.)
    print(f"[CDVAE OPTIMIZATION] target_bg: {target_bg}")
    print(f"[CDVAE OPTIMIZATION] coef_100more: {coef_100more}")
    print(f"[CDVAE OPTIMIZATION] coef_tolerance: {coef_tolerance}")
    print(f"[CDVAE OPTIMIZATION] num_gradient_steps: {num_gradient_steps}")
    print(f"[CDVAE OPTIMIZATION] MEGNET loss: {megnet_loss_mode}")

    opt = Adam([z], lr=lr)
    model.freeze()

    all_crystals = []
    if num_saved_crys <= 1:
        save_last_only = True
    else:
        save_last_only = False
        interval = num_gradient_steps // (num_saved_crys-1)

    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()

        preds = _raw_property_preds(model, z)   # shape = (N,4) or (N,1)
        # 🔽 MEGNet モードなら、4つの出力を解釈して個別に損失構成
        if megnet_loss_mode:
            pred_gap, pred_eform, pred_100more, pred_tolerance = preds.T

            # 🔽 標準引数 target_bg を args から取得（main() の args を optimization() に渡す必要あり）
            target_bg = getattr(ld_kwargs, 'target_bg', -1.)  # ← 無指定なら None
            if target_bg < 0.:
                raise ValueError("target_bg must be specified for MEGNet mode")

            loss = (
                - coef_100more*pred_100more.mean()
                - coef_tolerance*pred_tolerance.mean()
                + coef_e_form*pred_eform.mean()
                + torch.clip(torch.abs(pred_gap - target_bg) - 0.04, min=0.0).mean()
            )
        else:
            # 通常モード（superconなど）
            loss = sign * preds.mean()         # supercon など従来通り

        loss.backward()
        opt.step()

        #if (save_last_only and i == num_gradient_steps - 1) or (not save_last_only and (i % interval == 0 or i == num_gradient_steps - 1)):
        #    print(f'Current step:{i}.')
        #    crystals = model.langevin_dynamics(z, ld_kwargs)
        #    all_crystals.append(crystals)
        # ─────────────────────────────────────────────
        # ★ 1 結晶ずつ langevin_dynamics を回す ★
        # ─────────────────────────────────────────────
        if (save_last_only and i == num_gradient_steps - 1) or (not save_last_only and (i % interval == 0 or i == num_gradient_steps - 1)):
                print(f"Current step:{i}.  sampling z_batch size = {z.shape[0]}")
                merged = _sample_batch(model, ld_kwargs, z)      # ← ここだけで OK
                all_crystals.append(merged)

    #return {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
    #        ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}
    #result = {
    #    k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0)
    #    for k in ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']
    #}
    result = {                                   # shape = [save_step, N*, ...]
        k: torch.stack([torch.tensor(d[k]) if isinstance(d[k], float) else d[k]
                        for d in all_crystals], dim=0)
        for k in ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']
    }

    # 🔽 🔽 🔽 ここに①と②を入れる 🔽 🔽 🔽
    print("[DEBUG] result shapes:")
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    total_atoms = result['num_atoms'][0].sum().item()
    frac_coords_count = result['frac_coords'].shape[1]
    print(f"[DEBUG] total_atoms (from num_atoms): {total_atoms}")
    print(f"[DEBUG] frac_coords.shape[1]:         {frac_coords_count}")

    # 🔽 推論結果を追加：z に対する予測値
    with torch.no_grad():
        # 元の z に対する予測値
        #preds = model.fc_property(z).detach().cpu()
        #result['prediction'] = preds.squeeze(1)
        preds = _raw_property_preds(model, z).detach().cpu()  # ← 修正
        result['prediction'] = preds              # (N,4) or (N,1)

        # 全ての構造をまとめて1回でデコードする
        crystal_list = []
        # ──────────────────────────────────────────────────────────
        # ここでフラットな配列を「結晶ごと」に切り分ける
        # ──────────────────────────────────────────────────────────
        frac_coords_flat = result['frac_coords'][0].cpu()      # [総原子数, 3]
        atom_types_flat  = result['atom_types'][0].cpu()        # [総原子数]
        lengths_all      = result['lengths'][0].cpu()           # [結晶数, 3]
        angles_all       = result['angles'][0].cpu()            # [結晶数, 3]
        num_atoms_all    = result['num_atoms'][0].cpu().tolist()# [結晶数]

        crystal_list = []
        parent_indices = []       # ★追加：親 z のインデックス

        ptr = 0
        for i, n in enumerate(num_atoms_all):
            n = int(n)              # tensor → int
            if n == 0:              # 念のため
                continue

            fc = frac_coords_flat[ptr:ptr+n]        # [n, 3]
            at = atom_types_flat[ptr:ptr+n]         # [n]
            ptr += n

            crystal_dict = {
                'frac_coords': fc,                  # tensor(float32) で OK
                'atom_types' : at,                  # tensor(int64)   で OK
                'lengths'    : lengths_all[i],      # tensor(3)
                'angles'     : angles_all[i],       # tensor(3)
                'num_atoms'  : n,
            }
            crystal_list.append(crystal_dict)
            parent_indices.append(i % num_starting_points)

        print(f"[DEBUG] crystal_list length (after regroup) = {len(crystal_list)}")

        # 🔽🔽🔽 ここに追加 🔽🔽🔽
        def _is_reasonable(lengths, angles):
            return (lengths > 0).all() and (angles > 0).all() and (angles < 180).all()

        valid_crystals = []
        valid_parent_indices = []
        for d, idx in zip(crystal_list, parent_indices):
            if _is_reasonable(d['lengths'], d['angles']):
                valid_crystals.append(d)
                valid_parent_indices.append(idx)
            else:
                print("🚫 skip unrealistic lattice:", d['lengths'], d['angles'])

        if len(valid_crystals) == 0:
            print("🛑 All sampled structures were filtered out.")
            return result
        crystal_list = valid_crystals
        parent_indices = valid_parent_indices
        # 🔼🔼🔼 ここまで追加 🔼🔼🔼

        dataset = TensorCrystDataset(
            crystal_list,
            niggli=model.hparams.data.niggli,
            primitive=model.hparams.data.primitive,
            graph_method=model.hparams.data.graph_method,
            lattice_scale_method=model.hparams.data.lattice_scale_method,
            preprocess=False,
            preprocess_workers=0,  # 並列数は任意で調整
        )
        dataset.scaler = model.scaler
        dataset.lattice_scaler = model.lattice_scaler

        # 🔽🔽ここに⑤を挿入🔽🔽
        print(f"[DEBUG] Dataset[0] keys: {dataset[0].__dict__.keys()}")

        # Dataset の長さが「前処理を通った結晶数」
        parent_indices = torch.tensor(parent_indices)[:len(dataset)]

        batch = Batch.from_data_list(dataset).to(model.device)
        print(f"[DEBUG] batch size: {batch.num_graphs}")
        _, _, decoded_z = model.encode(batch)

        #preds_decoded = model.fc_property(decoded_z).detach().cpu()
        #result['prediction_decoded'] = preds_decoded.squeeze(1)
        preds_decoded = _raw_property_preds(model, decoded_z).detach().cpu()
        result['prediction_decoded'] = preds_decoded

        # ★ prediction を同じ長さにそろえて格納
        #result['prediction_matched'] = preds[parent_indices].squeeze(1)
        result['prediction_matched'] = preds[parent_indices]

        # 例：誤差を確認
        diff = (result['prediction_matched'] - result['prediction_decoded']).abs()
        print("MAE =", diff.mean().item())

    return result

def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=('recon' in args.tasks) or
        ('opt' in args.tasks and args.start_from == 'data'))
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar,
                                target_bg=args.target_bg, # 🔽 追加
                                )

    if torch.cuda.is_available():
        model.to('cuda')

    if 'recon' in args.tasks:
        print('Evaluate model on the reconstruction task.')
        start_time = time.time()
        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack, input_data_batch) = reconstructon(
            test_loader, model, ld_kwargs, args.num_evals,
            args.force_num_atoms, args.force_atom_types, args.down_sample_traj_step)

        if args.label == '':
            recon_out_name = 'eval_recon.pt'
        else:
            recon_out_name = f'eval_recon_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / recon_out_name)

    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        start_time = time.time()

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = generation(
            model, ld_kwargs, args.num_batches_to_samples, args.num_evals,
            args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_gen.pt'
        else:
            gen_out_name = f'eval_gen_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / gen_out_name)

    if 'opt' in args.tasks:
        print('Evaluate model on the property optimization task.')
        start_time = time.time()
        if args.start_from == 'data':
            loader = test_loader
        else:
            loader = None
        optimized_crystals = optimization(
            model, ld_kwargs, loader,lr=args.lr,num_starting_points=args.num_starting_points, 
            num_gradient_steps=args.num_gradient_steps,num_saved_crys=args.num_saved_crys,
            megnet_loss_mode=args.megnet_loss_mode,
            coef_e_form=args.coef_e_form,           # ← 追加
            coef_100more=args.coef_100more,         # ← 追加
            coef_tolerance=args.coef_tolerance      # ← 追加
        )
        optimized_crystals.update({'eval_setting': args,
                                   'time': time.time() - start_time})

        if args.label == '':
            gen_out_name = f'eval_opt_bg{args.target_bg:.2f}.pt'
        else:
            gen_out_name = f'eval_opt_{args.label}__bg{args.target_bg:.2f}__lr{args.lr}__grad-steps{args.num_gradient_steps}.pt'
        torch.save(optimized_crystals, model_path / gen_out_name)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num_starting_points', default=100, type=int)
    parser.add_argument('--num_gradient_steps', default=5000, type=int)
    parser.add_argument('--num_saved_crys', default=10, type=int)
    parser.add_argument('--label', default='')
    parser.add_argument('--megnet_loss_mode', default=False, type=bool)
    parser.add_argument('--target_bg', default=-1., type=float,
                    help='Target bandgap value used in optimization loss (megnet mode)')
    parser.add_argument('--coef_e_form',  default=1.0, type=float,
                        help='Loss coefficient for e_form (0 で無効)')
    parser.add_argument('--coef_100more',  default=0.0, type=float,
                        help='Loss coefficient for 100more (0 で無効)')
    parser.add_argument('--coef_tolerance', default=0.0, type=float,
                        help='Loss coefficient for tolerance (0 で無効)')

    args = parser.parse_args()

    main(args)
