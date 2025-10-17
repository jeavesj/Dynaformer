import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
from preprocess import gen_feature, gen_graph, to_pyg_graph, get_info, RF_score, GB_score, GetECIF
from joblib import Parallel, delayed
from utils import read_mol, obabel_pdb2mol, pymol_pocket
import numpy as np
from rdkit import Chem, RDLogger
import tempfile
import pandas as pd
import os
import time
import json

# global placeholders set in __main__ so process_one can write per-pose artifacts
PARTS_DIR = None
TIMING_DIR = None

def _now():
    return time.perf_counter()

def _safe_write_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

def _json_default(x):
    # make numpy things serializable
    try:
        import numpy as _np
        if isinstance(x, (_np.integer,)): return int(x)
        if isinstance(x, (_np.floating,)): return float(x)
        if isinstance(x, (_np.ndarray,)): return x.tolist()
        if isinstance(x, _np.dtype): return str(x)
    except Exception:
        pass
    return str(x)

def _safe_write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, default=_json_default)
    os.replace(tmp, path)

def process_one(proteinpdb: Path, ligandsdf: Path, name: str, pk: float, protein_cutoff, pocket_cutoff, spatial_cutoff):
    RDLogger.DisableLog('rdApp.*')

    t0 = _now()
    timing = {'name': name, 'status': 'ok', 'fail_reason': None}
    out_graph_path = Path(PARTS_DIR) / f'{name}.pkl'
    out_timing_path = Path(TIMING_DIR) / f'{name}.timing.json'

    if not (proteinpdb.is_file() and ligandsdf.is_file()):
        print(f'{proteinpdb} or {ligandsdf} does not exist.')
        timing['status'] = 'fail'
        timing['fail_reason'] = 'missing_input'
        timing['t_total_s'] = _now() - t0
        _safe_write_json(timing, out_timing_path)
        return None

    pocketpdb = proteinpdb.parent / (proteinpdb.name.rsplit('.', 1)[0] + '_pocket.pdb')
    pocketsdf = proteinpdb.parent / (proteinpdb.name.rsplit('.', 1)[0] + '_pocket.sdf')

    t = _now()
    if not pocketpdb.is_file():
        try:
            pymol_pocket(proteinpdb, ligandsdf, pocketpdb)
        except Exception as e:
            timing['status'] = 'fail'
            timing['fail_reason'] = f'pymol_pocket_error: {e}'
            timing['t_make_pocket_s'] = _now() - t
            timing['t_total_s'] = _now() - t0
            _safe_write_json(timing, out_timing_path)
            return None
    timing['t_make_pocket_s'] = _now() - t

    t = _now()
    if not pocketsdf.is_file():
        try:
            obabel_pdb2mol(pocketpdb, pocketsdf)
        except Exception as e:
            timing['status'] = 'fail'
            timing['fail_reason'] = f'obabel_pdb2mol_error: {e}'
            timing['t_pocket_convert_s'] = _now() - t
            timing['t_total_s'] = _now() - t0
            _safe_write_json(timing, out_timing_path)
            return None
    timing['t_pocket_convert_s'] = _now() - t

    try:
        t = _now()
        ligand = read_mol(ligandsdf)
        pocket = read_mol(pocketsdf)
        timing['t_read_mols_s'] = _now() - t

        t = _now()
        proinfo, liginfo = get_info(proteinpdb, ligandsdf)
        timing['t_get_info_s'] = _now() - t

        t = _now()
        res = gen_feature(ligand, pocket, name)
        timing['t_gen_feature_s'] = _now() - t

        t = _now()
        res['rfscore'] = RF_score(liginfo, proinfo)
        res['gbscore'] = GB_score(liginfo, proinfo)
        res['ecif'] = np.array(GetECIF(str(proteinpdb), str(ligandsdf)))
        timing['t_classic_scores_s'] = _now() - t
    except RuntimeError:
        print(proteinpdb, pocketsdf, ligandsdf, 'Fail on reading molecule')
        timing['status'] = 'fail'
        timing['fail_reason'] = 'read_or_feature_error'
        timing['t_total_s'] = _now() - t0
        _safe_write_json(timing, out_timing_path)
        return None
    except Exception as e:
        timing['status'] = 'fail'
        timing['fail_reason'] = f'prep_error: {e}'
        timing['t_total_s'] = _now() - t0
        _safe_write_json(timing, out_timing_path)
        return None

    ligand_tuple = (res['lc'], res['lf'], res['lei'], res['lea'])
    pocket_tuple = (res['pc'], res['pf'], res['pei'], res['pea'])
    try:
        t = _now()
        raw = gen_graph(ligand_tuple, pocket_tuple, name, protein_cutoff=protein_cutoff, pocket_cutoff=pocket_cutoff, spatial_cutoff=spatial_cutoff)
        timing['t_gen_graph_s'] = _now() - t
    except ValueError as e:
        print(f'{name}: Error gen_graph from raw feature {str(e)}')
        timing['status'] = 'fail'
        timing['fail_reason'] = f'gen_graph_error: {e}'
        timing['t_total_s'] = _now() - t0
        _safe_write_json(timing, out_timing_path)
        return None

    t = _now()
    graph = to_pyg_graph(list(raw) + [res['rfscore'], res['gbscore'], res['ecif'], pk, name], frame=-1, rmsd_lig=0.0, rmsd_pro=0.0)
    timing['t_to_pyg_s'] = _now() - t
    timing['t_total_s'] = _now() - t0
    
    # Capture node/edge stats
    timing['n_nodes'] = int(graph.num_nodes)
    try:
        timing['n_edges'] = int(graph.num_edges)
    except Exception:
        try:
            timing['n_edges'] = graph.edge_index.shape[1]
        except Exception:
            timing['n_edges'] = None
    try:
        timing['edge_type_max'] = int(graph.edge_type.max())
    except Exception:
        timing['edge_type_max'] = None

    # per-pose artifacts
    timing['status'] = 'ok'
    _safe_write_pickle(graph, out_graph_path)
    _safe_write_json(timing, out_timing_path)

    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_csv', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--protein_cutoff', type=float, default=5.0)
    parser.add_argument('--pocket_cutoff', type=float, default=5.0)
    parser.add_argument('--spatial_cutoff', type=float, default=5.0)
    parser.add_argument('--skip_crystals', action='store_true')
    
    args = parser.parse_args()
    filedf = pd.read_csv(args.file_csv)
    receptors = filedf['receptor']
    ligands = filedf['ligand']
    names = filedf['name']
    try:
        pks = filedf['pk']
    except:
        filedf['pk'] = -1
        pks = filedf['pk']


    # derive per-pose directories from the requested output path
    base = args.output
    if base.suffix:
        parts_dir = base.with_suffix('')
    else:
        parts_dir = base
    parts_dir = parts_dir.with_name(parts_dir.name + '.parts')
    timing_dir = parts_dir.with_name(parts_dir.name + '.timing')

    parts_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)

    # set globals used by worker
    PARTS_DIR = str(parts_dir)
    TIMING_DIR = str(timing_dir)

    graphs = Parallel(n_jobs=args.njobs)(
        delayed(process_one)(
            Path(rec), Path(lig), name, pk,
            args.protein_cutoff, args.pocket_cutoff, args.spatial_cutoff
        )
        for rec, lig, name, pk in zip(tqdm(receptors), ligands, names, pks)
    )
    graphs = list(filter(None, graphs))
    pickle.dump(graphs, open(args.output, 'wb'))
