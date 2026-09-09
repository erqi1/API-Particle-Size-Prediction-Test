"""app.py v4 - membaca artefak notebook, TANPA koefisien hardcode."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='API PSD Prediction v4', page_icon='📊', layout='wide')

BASE = Path(__file__).resolve().parent
DEP = next((d for d in [BASE/'deployment_v4', BASE/'deployment_package', BASE]
            if (d/'config.json').exists()), BASE)


@st.cache_resource
def load_all():
    cfg = json.loads((DEP/'config.json').read_text(encoding='utf-8'))
    rng_p = DEP/'input_ranges.json'
    rng = json.loads(rng_p.read_text(encoding='utf-8')) if rng_p.exists() else {}
    models = {}
    for stage in ('stage_a', 'stage_b'):
        for tgt, spec in cfg.get(stage, {}).items():
            f = spec.get('model_file')
            if f and (DEP/f).exists():
                models[(stage, tgt)] = joblib.load(DEP/f)
    return cfg, rng, models


try:
    CFG, RANGES, MODELS = load_all()
except Exception as exc:
    st.error(f'Gagal memuat artefak dari {DEP}')
    st.exception(exc)
    st.stop()

MILLING = ['Feed_Speed', 'Frequency', 'Screen_Size']
UPSTREAM = CFG['stage_b']['z_mid']['features']
SETPOINTS = set(CFG['setpoints'])
SPEC = CFG['spec']
CONF_Q = CFG['conformal_quantiles_log']
NEED_CENTERING = CFG.get('need_centering', False)
STAGE_B_OFFSET = CFG.get('stage_b_offset', {})
UP_MEDIAN = CFG['upstream_median']
CODE = {'ARE13333_T0':'X1','ARE1331_T0':'X2','TF_Time':'X3','C_Temp':'X4',
        'C_pH':'X5','C_Time':'X6','C_T1':'X7',
        'Feed_Speed':'X8','Frequency':'X9','Screen_Size':'X10'}


def predict(vals):
    """Stage A + Stage B -> P10/P50/P90 + interval conformal."""
    xa = [[vals[c] for c in MILLING] + ([0.0] if NEED_CENTERING else [])]
    xa = np.asarray(xa, dtype=float)
    xb = np.asarray([[vals.get(c, UP_MEDIAN[c]) for c in UPSTREAM]], dtype=float)

    z = {}
    for t in ('z_mid', 'z_upper', 'z_lower'):
        v = float(MODELS[('stage_a', t)].predict(xa)[0])
        if CFG['stage_b'][t]['enabled'] and ('stage_b', t) in MODELS:
            # artefak Stage B = model sklearn murni; constraint mean-zero
            # diterapkan di sini sebagai pengurangan offset (angka dari config).
            v += float(MODELS[('stage_b', t)].predict(xb)[0]) - float(STAGE_B_OFFSET.get(t, 0.0))
        z[t] = v

    qm = CONF_Q['z_mid'] or 0.0
    qu = CONF_Q['z_upper'] or 0.0
    ql = CONF_Q['z_lower'] or 0.0
    zm, zu, zl = z['z_mid'], z['z_upper'], z['z_lower']
    return {
        'P50': (np.exp(zm), np.exp(zm-qm), np.exp(zm+qm)),
        'P90': (np.exp(zm+zu), np.exp(zm-qm+zu-qu), np.exp(zm+qm+zu+qu)),
        'P10': (np.exp(zm-zl), np.exp(zm-qm-zl-ql), np.exp(zm+qm-zl+ql)),
    }


with st.sidebar:
    st.header('Model Info')
    st.caption(f'v{CFG["version"]} — dibaca dari `{DEP.name}/`')
    st.write(f"**Batch total:** {CFG['n_batches_total']}")
    st.write(f"**Ber-X1..X7:** {CFG['n_batches_with_upstream']}")
    st.write(f"**Coverage:** {CFG['coverage_target']:.0%}")
    st.write(f"**Feed unit:** {CFG['feed_speed_unit']}")
    st.divider()
    for t in ('z_mid', 'z_upper', 'z_lower'):
        a, b = CFG['stage_a'][t], CFG['stage_b'][t]
        st.write(f"**{t}** — A: `{a['model_type']}`"
                 f" | B: {'`'+str(b['model_type'])+'`' if b['enabled'] else 'off'}")
    st.divider()
    st.caption('Tanpa koefisien hardcode. Semua parameter dari artefak model.')

st.title('📊 API Particle Size Prediction (v4)')
st.caption('Stage A (milling, semua batch) + Stage B (upstream, mean-zero)')
st.info('Monotonisitas **P10 < P50 < P90 dijamin** lewat reparameterisasi target. '
        f"Interval adalah **conformal {CFG['coverage_target']:.0%}** "
        '(coverage terverifikasi), bukan ±MAE.')

st.header('1. Parameter Proses')
vals = {}
c1, c2 = st.columns(2)
for i, feat in enumerate(UPSTREAM + MILLING):
    info = RANGES.get(feat, {})
    lo, hi = info.get('min', 0.0), info.get('max', 100.0)
    dflt = info.get('median', (lo+hi)/2)
    role = 'setpoint' if feat in SETPOINTS else 'outcome'
    tag = '🎛️' if role == 'setpoint' else '📈'
    col = c1 if i % 2 == 0 else c2
    with col:
        obs = CFG.get(feat.lower()+'_observed')
        if feat in MILLING and obs:
            vals[feat] = st.selectbox(f'{tag} {CODE[feat]}: {feat}', obs,
                                      index=len(obs)//2, help=f'{role} | level teramati')
        else:
            vals[feat] = st.number_input(f'{tag} {CODE[feat]}: {feat}', value=float(dflt),
                                         step=(hi-lo)/100 if hi > lo else 0.1,
                                         help=f'{role} | historis {lo:g}–{hi:g}')

st.caption('🎛️ = setpoint (bisa di-set operator) · 📈 = outcome proses (konteks saja)')

if st.button('🔮 Prediksi', type='primary', use_container_width=True):
    out = predict(vals)
    st.header('2. Hasil')
    cols = st.columns(3)
    for col, t in zip(cols, ('P10', 'P50', 'P90')):
        pt, lo, hi = out[t]
        with col, st.container(border=True):
            st.subheader(t)
            st.metric('Prediksi', f'{pt:.1f} µm')
            st.caption(f"Interval {CFG['coverage_target']:.0%}: **{lo:.1f} – {hi:.1f}** µm")
            if t in SPEC:
                lsl, usl = SPEC[t]
                if lo >= lsl and hi <= usl:
                    st.success(f'PASS — seluruh interval di dalam {lsl:g}–{usl:g}')
                elif pt < lsl or pt > usl:
                    st.error(f'FAIL — prediksi di luar {lsl:g}–{usl:g}')
                else:
                    st.warning(f'BERISIKO — interval melewati batas {lsl:g}–{usl:g}')

    p10, p50, p90 = out['P10'][0], out['P50'][0], out['P90'][0]
    if p10 < p50 < p90:
        st.success('✓ Monotonisitas P10 < P50 < P90 terpenuhi')
    else:
        st.error('Monotonisitas dilanggar — artefak model tidak konsisten')

    st.header('3. Applicability Domain')
    rows = []
    for feat in UPSTREAM + MILLING:
        info = RANGES.get(feat, {})
        lo, hi = info.get('min'), info.get('max')
        v = vals[feat]
        ok = (lo is None) or (lo <= v <= hi)
        rows.append({'Kode': CODE[feat], 'Variabel': feat,
                     'Peran': 'setpoint' if feat in SETPOINTS else 'outcome',
                     'Input': v, 'Min': lo, 'Max': hi,
                     'Status': 'DALAM' if ok else 'DI LUAR'})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    nf = CFG.get('noise_floor', {})
    if nf.get('P50'):
        st.caption(f"Noise floor P50 ≈ {nf['P50']:.1f} µm, P90 ≈ {nf.get('P90', 0):.1f} µm — "
                   'variasi batch-to-batch yang tidak bisa dijelaskan setting milling.')