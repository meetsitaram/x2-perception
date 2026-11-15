import json, sys
R = '/mnt/filesystem-g4/tracks/acp/runs/'
def load(p):
    out = []
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out
s2 = load(R + 's2_v5/metrics.jsonl')
s3 = load(R + 's3_v2/metrics.jsonl')
s2 = [d for d in s2 if d.get('it', 0) <= 19218]
sets = sorted(k.split('/', 1)[1] for k in s3[-1] if k.startswith('finish/'))
ref = [s for s in sets if s != 'goal_tiles']

def agg(rows):
    o = {}
    E = F = 0.0
    for s in sets:
        e = sum(d.get('episodes/' + s, 0) for d in rows)
        f = sum(d.get('episodes/' + s, 0) * d.get('finish/' + s, 0) for d in rows)
        o[s] = (f / e if e else float('nan'), e)
        if s != 'goal_tiles': E += e; F += f
    o['REF_ALL'] = (F / E if E else float('nan'), E)
    def m(k):
        v = [d[k] for d in rows if k in d]
        return sum(v) / len(v) if v else float('nan')
    for k in ['end/finished_phase099', 'end/term_diverged', 'end/term_base_pos_gt_0.5m', 'end/term_base_rot_gt_1rad',
              'end/timeout_before_phase099', 'goal/success', 'goal/fall', 'goal/level_mean', 'goal/waypoint_index_mean',
              'planner/replay_size', 'planner/updates', 'ppo/value_loss', 'ppo/value_loss_goal', 'ppo/kl', 'ppo/action_std',
              'reward_per_step', 'env_steps_per_s', 'stage2/eta_obs', 'stage2/eta_rew', 'stage3/warmup', 'raw/track_joint_pos',
              'raw/track_base_pos', 'err/joint_pos_rad_mean_abs', 'err/base_pos_m', 'episodes', 'episodes/goal_tiles',
              'raw/goal/move2goal', 'raw/goal/standatgoal', 'planner/recorded_successes', 'planner/recorded_episodes']:
        o[k] = m(k)
    o['it'] = (rows[0]['it'], rows[-1]['it'])
    return o

def show(name, o):
    print('==', name, o['it'])
    print(' REF_ALL %.4f (n=%d)  goal_tiles %.4f (n=%d)' % (o['REF_ALL'] + o['goal_tiles']))
    for k in ['end/finished_phase099', 'end/term_diverged', 'end/timeout_before_phase099', 'end/term_base_pos_gt_0.5m', 'end/term_base_rot_gt_1rad', 'goal/success', 'goal/fall', 'goal/level_mean', 'goal/waypoint_index_mean', 'planner/replay_size', 'planner/updates', 'ppo/value_loss', 'ppo/value_loss_goal', 'ppo/kl', 'ppo/action_std', 'reward_per_step', 'env_steps_per_s', 'stage2/eta_obs', 'stage2/eta_rew', 'stage3/warmup', 'raw/track_joint_pos', 'raw/track_base_pos', 'err/joint_pos_rad_mean_abs', 'err/base_pos_m', 'episodes', 'episodes/goal_tiles', 'raw/goal/move2goal', 'raw/goal/standatgoal']:
        print('  %-32s %.4f' % (k, o[k]))

A = agg(s2[-300:]); show('s2_v5 last 300', A)
W = [s3[i:i + 100] for i in range(0, len(s3), 100)]
Bs = [agg(w) for w in W]
for b in Bs: show('s3 window', b)
print('\nper set: s2_end(300) | s3 windows of 100 rows  (finish, episodes in last)')
for s in ref + ['goal_tiles']:
    print('  %-44s %.3f | ' % (s, A[s][0]) + ' '.join('%.3f' % b[s][0] for b in Bs) + '  n=%d' % Bs[-1][s][1])
# first rows of s3 (continuity)
print('\nfirst 5 s3 rows')
for d in s3[:5]:
    print(' it', d['it'], 'eta', d.get('stage2/eta_obs'), d.get('stage2/eta_rew'), 'warm', d.get('stage3/warmup'), 'ref_fin_flat', d.get('finish/x2_loco_flat_train_hold'), 'goal_succ', d.get('goal/success'), 'lvl', d.get('goal/level_mean'), 'div', d.get('end/term_diverged'), 'vl', d.get('ppo/value_loss'), d.get('ppo/value_loss_goal'), 'std', d.get('ppo/action_std'), 'replay', d.get('planner/replay_size'))
print('last s2 row it', s2[-1]['it'], 'std', s2[-1].get('ppo/action_std'), 'div', s2[-1].get('end/term_diverged'), 'replay', s2[-1].get('planner/replay_size'))
# NaN scan
bad = [(d['it'], k) for d in s3 for k, v in d.items() if isinstance(v, float) and v != v]
print('NaN entries in s3:', len(bad), bad[:10])
# goal levels per family over windows
fam = sorted(k for k in s3[-1] if k.startswith('goal/level/'))
print('\ngoal levels (first row, window means)')
for k in fam:
    print('  %-30s %.2f | ' % (k, s3[0].get(k, float('nan'))) + ' '.join('%.2f' % (sum(d.get(k, 0) for d in w) / len(w)) for w in W))
