#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors for the prompt-excluded release; numerical/parser/guard scope is documented in docs/REPRODUCTION_SCOPE.md.
"""Recompute selected reported results from bundled exports, offline and read-only."""
from __future__ import annotations

import collections
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics as st
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis'))
sys.path.insert(0, str(ROOT/'application'))
from _existing_statistics import mean_ci, mixtures, welch
from _router_metrics import scores
from phone_agent.fast_path import FastPathRouter

DATA, EXPECTED = ROOT/'data', ROOT/'expected'
CHECKS = []
LAYERS = {'easy':'Simple', 'normal':'Normal', 'complex':'Complex', 'refusal':'Refusal'}
MODES = {'Hybrid':'Routed', 'Slow-only':'Cloud-only'}
DEVICES = ['High', 'Mid', 'Low']
CONDITIONS = {'C1':('Launch','Routed',12), 'C2':('Launch','Cloud-only',12),
              'C3':('Workflow','Routed',6), 'C4':('Workflow','Cloud-only',6)}


def load(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scalar(value):
    if value == '':
        return None
    if value in {'True', 'False', 'true', 'false'}:
        return value.lower() == 'true'
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return [{key:scalar(value) for key,value in row.items()} for row in csv.DictReader(stream)]


def equivalent(actual, expected, location='root'):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (location, set(actual)^set(expected))
        for key in expected:
            equivalent(actual[key], expected[key], location+'.'+str(key))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), (location, len(actual), len(expected))
        for index, (a,b) in enumerate(zip(actual,expected)):
            equivalent(a,b,location+f'[{index}]')
    elif isinstance(expected, (int,float)) and not isinstance(expected,bool):
        assert math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-9), (location,actual,expected)
    else:
        assert actual == expected, (location,actual,expected)


def compare_csv(name, actual):
    equivalent(actual, read_csv(EXPECTED/name), name)
    CHECKS.append(dict(output=name, rows=len(actual), matched=True))


def fourlayer():
    rows = lines(DATA/'fourlayer_slots.jsonl')
    assert len(rows) == 504 and len({(r['device'],r['run_id']) for r in rows}) == 504
    summaries = []
    for device in DEVICES:
        sel = [r for r in rows if r['device']==device]
        summaries.append(dict(device=device, planned=len(sel),
            judgeable=sum(r['success'] is not None for r in sel),
            unjudgeable=sum(r['success'] is None for r in sel),
            **{'pass':sum(r['success'] is True for r in sel)},
            fail=sum(r['success'] is False for r in sel),
            timeout_failures_retained=sum(r['machine']=='timeout' for r in sel)))
    compare_csv('A_denominators.csv',summaries)
    boundaries, common = [], []
    for device in DEVICES+['All']:
        selected = [r for r in rows if device=='All' or r['device']==device]
        for layer in ['All']+list(LAYERS):
            for mode in ['All','Hybrid','Slow-only']:
                sel = [r for r in selected if (layer=='All' or r['layer']==layer) and (mode=='All' or r['mode']==mode)]
                n, passed = len(sel), sum(r['success'] is True for r in sel)
                unknown = sum(r['success'] is None for r in sel)
                boundaries.append(dict(device=device,layer=LAYERS.get(layer,layer),mode=MODES.get(mode,mode),
                    planned_n=n,judged_n=n-unknown,**{'pass':passed},unjudgeable=unknown,
                    primary_rate_pct=100*passed/(n-unknown),all_unjudgeable_fail_n=n,
                    all_fail_pass=passed,all_fail_rate_pct=100*passed/n,
                    all_success_pass=passed+unknown,all_success_rate_pct=100*(passed+unknown)/n))
        blocked = {(r['device'],r['task_id'],r['state']) for r in selected if r['success'] is None}
        kept = [r for r in selected if (r['device'],r['task_id'],r['state']) not in blocked]
        for layer in ['All']+list(LAYERS):
            for mode in ['Hybrid','Slow-only']:
                sel = [r for r in kept if (layer=='All' or r['layer']==layer) and r['mode']==mode]
                passed = sum(r['success'] is True for r in sel)
                common.append(dict(device=device,layer=LAYERS.get(layer,layer),mode=MODES[mode],
                    common_judged_n=len(sel),**{'pass':passed},rate_pct=100*passed/len(sel)))
    compare_csv('A_success_boundaries.csv',boundaries)
    compare_csv('A_common_judgeable.csv',common)


def energy():
    observations = read_csv(DATA/'energy_pairs.csv')
    assert len(observations)==36
    stats, contrasts = [], []
    for device in DEVICES:
        for condition,(workload,mode,count) in CONDITIONS.items():
            selected = [r for r in observations if r['device']==device and r['condition']==condition]
            stat = mean_ci([r['net_mah_batch'] for r in selected])
            stats.append(dict(device=device,condition=condition,workload=workload,mode=mode,
                unit='mAh/batch',requests_per_batch=count,**stat,mean_mah_request=stat['mean']/count,
                ci95_low_mah_request=stat['ci95_low']/count,ci95_high_mah_request=stat['ci95_high']/count))
        for workload,count in [('Launch',12),('Workflow',6)]:
            groups = [[r['net_mah_batch'] for r in observations if r['device']==device
                       and r['workload']==workload and r['mode']==mode] for mode in ['Routed','Cloud-only']]
            contrast = welch(*groups)
            contrasts.append(dict(device=device,workload=workload,n_routed=3,n_cloud=3,
                unit='mAh/batch',contrast='Routed minus Cloud-only',**contrast,
                difference_mah_request=contrast['difference']/count,
                ci95_low_mah_request=contrast['ci95_low']/count,ci95_high_mah_request=contrast['ci95_high']/count))
    compare_csv('B_energy_mean_t_ci.csv',stats)
    compare_csv('B_energy_difference_welch_ci.csv',contrasts)
    # Reuse the retained category-mixture functions; only temporary CSV outputs are written.
    with tempfile.TemporaryDirectory(prefix='dualroute-analysis-') as folder:
        mixtures(observations,Path(folder),load(DATA/'analysis_contract.json'))
        for name in ['C_category_cost_differences.csv','C_mixture_25_50_75.csv','C_conditional_zero_crossings.csv']:
            compare_csv(name,read_csv(Path(folder)/name))


def sensitivity():
    cells = read_csv(DATA/'window_replay_cells.csv')
    assert len(cells)==144
    groups = collections.defaultdict(list)
    for row in cells:
        groups[row['config_id'],row['profile']].append(row)
    summary = []
    fields = {'switch_count':'switch_count','degraded_step_count':'degraded_step_count',
              'detect_delay':'detect_delay_steps','recover_delay':'recover_delay_steps'}
    for (config,profile),sel in groups.items():
        assert len(sel)==4
        row = dict(config_id=config,profile=profile,noise_trace_count=4,
            expected_met_trace_count=sum(r['expected_trajectory_met'] is True for r in sel),
            false_trigger_count_total=sum(r['false_trigger_count'] for r in sel))
        for name,field in fields.items():
            values = [r[field] for r in sel if r[field] is not None]
            row[name+'_observed_n'] = len(values)
            for suffix,func in [('mean',st.mean),('min',min),('max',max)]:
                row[name+'_'+suffix] = func(values) if values else None
        summary.append(row)
    compare_csv('D_reaggregated_36_summaries.csv',summary)
    selected=[]
    for setting,config in [('P50','percentile_low'),('P75','baseline'),('P90','percentile_high'),
                           ('exit 5s','exit_ms_low'),('exit 8s','baseline'),('exit 11s','exit_ms_high')]:
        selected.append(dict(setting=setting,**next(r for r in summary if r['config_id']==config and r['profile']=='spike_recover')))
    compare_csv('D_percentile_exit_spike.csv',selected)


def budgets():
    rows = read_csv(DATA/'terminal_slots.csv')
    assert len(rows)==432 and sum(r['cap_censored'] for r in rows)==22
    result=[]
    for view,field in [('ledger_primary','ledger_seconds'),('raw_timing_sensitivity','terminal_seconds')]:
        for device in DEVICES+['All']:
            for role in ['Slow','Reject']:
                for mode in ['All','Routed','Cloud-only']:
                    for layer in ['All']+list(LAYERS.values()):
                        sel=[r for r in rows if (device=='All' or r['device']==device) and r['timing_role']==role
                             and (mode=='All' or r['mode']==mode) and (layer=='All' or r['gold_layer']==layer)]
                        if not sel:
                            continue
                        measured=[r for r in sel if r[field] is not None]
                        for budget in [30,40,50,60]:
                            misses=[r for r in measured if r[field]>budget]
                            result.append(dict(timing_view=view,device=device,timing_role=role,gold_layer=layer,mode=mode,
                                budget_s=budget,planned_n=len(sel),measured_n=len(measured),missing_n=len(sel)-len(measured),
                                ledger_missing_n=sum(r['ledger_seconds'] is None for r in sel),
                                unjudgeable_timing_recovered_n=sum(r['timing_recovered_from_semantically_unjudgeable'] for r in measured) if view=='raw_timing_sensitivity' else 0,
                                exceed_n=len(misses),exceed_pct_observed=100*len(misses)/len(measured) if measured else None,
                                exceed_lower_count_planned=len(misses),exceed_upper_count_planned_missing_all_exceed=len(misses)+len(sel)-len(measured),
                                semantic_failures_retained_n=sum(r['success'] is False for r in sel),
                                unjudgeable_in_planned_n=sum(r['success'] is None for r in sel),
                                timeout_censored_n=sum(r['cap_censored'] for r in sel),
                                timeout_censored_exceed_n=sum(r['cap_censored'] for r in misses),
                                unknown_guard_n=sum(r['guard_path'] is None for r in sel),
                                correct_semantic_success_within_budget_n=sum(r['success'] is True and r[field]<=budget for r in measured)))
    compare_csv('E_budget_exceedance.csv',result)


def router_latency():
    rows=read_csv(DATA/'router_call_latency.csv')
    actual={}
    for device in DEVICES:
        sel=[r for r in rows if r['device']==device]
        assert len(sel)==83
        values=[r['seconds'] for r in sel]
        actual[device]=dict(n=len(sel),distinct_tasks=len({r['task_id'] for r in sel}),
            app_states=dict(collections.Counter(r['state'] for r in sel)),median_s=st.median(values),
            mean_s=st.mean(values),min_s=min(values),max_s=max(values))
    equivalent(actual,load(EXPECTED/'router_latency_summary.json'))
    CHECKS.append(dict(output='router_latency_summary.json',rows=249,matched=True))


def saved_router():
    records=lines(DATA/'saved_router_predictions.jsonl')
    expected=load(EXPECTED/'main_app_replay_summary.json')
    assert len(records)==284
    for name,identity in expected['code_identity'].items():
        release = load(ROOT/'provenance/release_identity.json')
        assert release['research_core_identity'][name]['sha256'] == identity['sha256'], name
        release_hash = release['release_core_identity'][name]['sha256']
        assert hashlib.sha256((ROOT/'application'/name).read_bytes()).hexdigest() == release_hash, name
    summaries={}
    for method in ['cp800','baseline']:
        summary={}
        decisions={}
        for kind in ['base','variant']:
            sel=[r for r in records if r['input']['text_kind']==kind]
            gold=np.array([r['input']['gold_raw'] for r in sel])
            parsed,effective=[],[]
            for row in sel:
                assert hashlib.sha256(row['input']['text'].encode()).hexdigest()==row['text_sha256']
                saved=row['predictions'][method]
                decision=FastPathRouter(lambda text, raw=saved['raw']:raw).decide(row['input']['text'])
                assert decision.classifier_label==saved['parsed'] and decision.final_route==saved['final_route']
                assert decision.reason==saved['guard_reason']
                parsed.append(-1 if decision.classifier_label is None else int(decision.classifier_label))
                effective.append({'fast_path':0,'slow_path':1,'reject':2}[decision.final_route])
                decisions[row['input']['case_group_id'],kind]=(row['input']['gold_raw'],parsed[-1],effective[-1])
            p,e=np.array(parsed),np.array(effective)
            clar=np.array([r['input']['requires_clarification'] for r in sel])
            values=dict(parsed=scores(gold,p),effective=scores(gold,e),
                false_local_refusal=int(np.sum((gold!=2)&(e==2))),false_fast=int(np.sum((gold!=0)&(e==0))),
                reject_allowed_after_guard=int(np.sum((gold==2)&(e!=2))),guard_route_changes=int(np.sum(p!=e)),
                clarification=dict(n=int(clar.sum()),parsed_slow=int(np.sum(clar&(p==1))),
                    forwarded_slow=int(np.sum(clar&(e==1))),local_refusal=int(np.sum(clar&(e==2)))))
            for field,value in values.items():
                equivalent(value,expected['methods'][method][kind][field],method+'.'+kind+'.'+field)
            summary[kind]=values
        groups={key[0] for key in decisions}
        assert len(groups)==142
        for index,view in [(1,'parsed'),(2,'effective')]:
            count=sum(all(decisions[group,kind][index]==decisions[group,kind][0] for kind in ['base','variant']) for group in groups)
            key='both_texts_'+view+'_correct_groups'
            assert count==expected['methods'][method][key]
            summary[key]=count
        summaries[method]=summary
    CHECKS.append(dict(output='main_app_replay_summary.json',rows=284,matched=True,
        scope='main-app parser and full guard, point counts and metrics; stored bootstrap intervals and old-experiment-guard sensitivity not recomputed'))
    return summaries


def main():
    fourlayer()
    energy()
    sensitivity()
    budgets()
    router_latency()
    router=saved_router()
    print(json.dumps(dict(status='PASS_EXPORTED_DATA_RECOMPUTATION',checks=CHECKS,
        router_correct={method:{kind:{view:values[kind][view]['correct'] for view in ['parsed','effective']}
            for kind in ['base','variant']} for method,values in router.items()},
        network_calls=0,phone_calls=0,model_calls=0,
        limitations=['No independent raw screenshot adjudication or battery sensor reconstruction',
            'No inference, training, original experiment regeneration or full figure rebuild; mobile exports are verified separately',
            'Saved bootstrap intervals and historical mixed-study summaries retained as reference, not recomputed here']),
        ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
