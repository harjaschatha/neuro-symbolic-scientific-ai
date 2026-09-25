"""Portfolio adaptation of the research Streamlit workbench. Run: streamlit run app.py"""
import io
import json
import pandas as pd
import streamlit as st
from frontend.data import ROOT, RESULT_LABELS, load_replay, accuracy
from frontend import workbench as ui


@st.cache_data(show_spinner=False)
def replay(name):
    return load_replay(name)


def downloads(path, label, key):
    st.download_button(label, path.read_bytes(), file_name=path.name, key=key)


def render_results(name, rows):
    st.header('Frozen evaluation results')
    st.caption('Saved synthetic research runs. Each result set is kept separate.')
    counts = accuracy(rows)
    columns = st.columns(4)
    columns[0].metric('Cases', counts['total'])
    for col, metric, label in zip(columns[1:], ['saved', 'exact', 'family'], ['Saved projection-aware', 'Exact biological label', 'Fixed mechanism family']):
        col.metric(label, f"{counts[metric]/counts['total']:.1%}", help=f"{counts[metric]}/{counts['total']} cases")
    st.caption('Saved projection-aware targets differ between arms. Exact and fixed-family metrics use common labels; null biological truth means baseline.')
    if name == 'earlier_report':
        st.warning('Earlier report: 449/500 saved correct; original HPC provenance remains unresolved. A different local ablation copy records 376/500. This is not the Stage E cohort.')
    else:
        st.caption('Stage E: 500 paired seed-42 development cases. Run-time configuration and commit snapshots remain unverified.')
    options = ['Exact biological labels', 'Fixed mechanism families'] if name != 'earlier_report' else ['Original projected report matrix', 'Exact biological labels']
    choice = st.selectbox('Confusion matrix', options, key='matrix_metric')
    if choice == 'Original projected report matrix':
        base = ROOT/'figures/confusion/earlier_report'
        percent = st.checkbox('Show row percentages', key='normalized_matrix')
        png = base/('figure_5_2_projected_confusion_matrix_normalized.png' if percent else 'figure_5_2_projected_confusion_matrix.png')
        csv = base/('projected_fittable_confusion_matrix_row_percent.csv' if percent else 'projected_fittable_confusion_matrix_counts.csv')
        st.caption('Original report PNG, copied unchanged. Its count matrix matches the frozen earlier artifact: 449 diagonal entries out of 500. Switching truth is projected to maintenance.')
    else:
        metric = 'exact' if choice == 'Exact biological labels' else 'family'
        base = ROOT/f'figures/confusion/{name}'
        png, csv = base/f'{metric}_counts.png', base/f'{metric}_counts.csv'
        st.caption('Recomputed from the frozen case labels. Rows are ground truth; columns are selected mechanisms. Cells show counts.')
    st.image(str(png), width="stretch")
    left, right = st.columns(2)
    with left:
        downloads(png, 'Download matrix PNG', 'matrix_png')
    with right:
        downloads(csv, 'Download underlying matrix CSV', 'matrix_csv')
    with st.expander('Underlying matrix values'):
        st.dataframe(pd.read_csv(csv, index_col=0), width="stretch")
    st.subheader('Compare the paired Stage E arms')
    st.dataframe(pd.read_csv(ROOT/'results/verified/summary.csv').query("result_set.str.startswith('stage_e')", engine='python'), hide_index=True, width="stretch")
    st.caption('Expanded versus standard: +32.0 percentage points in exact accuracy, and +17.8 in fixed-family accuracy. Library coverage contributes to these differences.')


def render_samples():
    st.header('Synthetic data gallery')
    st.write('Original scenario figures and their CSV measurements from the main project.')
    manifest = json.loads((ROOT/'data/showcase_manifest.json').read_text())
    items = {item['name']: item for item in manifest['synthetic_samples']}
    selected = st.selectbox('Synthetic scenario and split', list(items), key='sample_name')
    item = items[selected]
    st.caption('These calibration/validation scenario illustrations are separate from the frozen 500-case evaluation cohorts.')
    st.image(str(ROOT/item['png']), width="stretch")
    df = pd.read_csv(ROOT/item['csv'])
    with st.expander('Explore measurements', expanded=True):
        available = [c for c in df.columns if c.endswith(('_obs', '_true', '_baseline')) or c.startswith('res_')]
        channels = st.multiselect('Channels', available, default=[c for c in ['X_obs', 'X_true', 'X_baseline'] if c in available], key='channels')
        if channels:
            st.line_chart(df.set_index('time_h')[channels])
        st.dataframe(df.head(12), hide_index=True, width="stretch")
    cols = st.columns(2)
    with cols[0]:
        downloads(ROOT/item['csv'], 'Download scenario CSV', 'sample_csv')
    with cols[1]:
        downloads(ROOT/item['png'], 'Download original figure', 'sample_png')
    st.subheader('CSV intake preview')
    st.caption('Explore the original representative upload sample or your own CSV. This preview does not perform inference.')
    use_sample = st.checkbox('Preview bundled representative upload sample', key='preview_sample')
    uploaded = st.file_uploader('Preview another CSV', type=['csv'], key='upload_csv')
    if uploaded is not None:
        ui.render_uploaded_csv(uploaded)
    elif use_sample:
        ui.render_uploaded_csv(io.BytesIO((ROOT/'data/synthetic/representative_upload_timeseries.csv').read_bytes()))
    downloads(ROOT/'data/synthetic/representative_upload_timeseries.csv', 'Download representative upload sample', 'representative_csv')


def render_cases(name, rows):
    st.header('Research workbench · saved case explorer')
    st.caption('The original frontend’s diagnostics, hypothesis, fitted-candidate, guard, report, and audit panels replay the saved research payloads.')
    if name == 'earlier_report':
        st.warning('Earlier report artifact: original HPC provenance unresolved. Saved projection correctness differs from exact biological correctness.')
    left, right = st.columns(2)
    mechanism = left.selectbox('Selected mechanism', ['All'] + sorted({r['best_mechanism'] for r in rows}), key='selected_mechanism')
    status = right.selectbox('Exact biological match', ['All', 'Correct', 'Incorrect'], key='exact_filter')
    search = st.text_input('Search run ID', key='case_search')
    filtered = [r for r in rows if (mechanism == 'All' or r['best_mechanism'] == mechanism)
                and (status == 'All' or ((r.get('true_mechanism') or 'baseline') == r['best_mechanism']) == (status == 'Correct'))
                and search.lower() in r['run_id'].lower()]
    if not filtered:
        st.info('No cases match these filters.')
        return
    by_id = {r['run_id']: r for r in filtered}
    chosen = st.selectbox('Saved case', list(by_id), key='case_id')
    row = by_id[chosen]
    exact = (row.get('true_mechanism') or 'baseline') == row['best_mechanism']
    st.caption(f"{len(filtered)} matching cases · Exact biological label: {'correct' if exact else 'incorrect'} · Saved projection-aware flag: {row['correct_selection']}")
    context = f"Study: {RESULT_LABELS[name]}. Frozen synthetic result replay. "
    context += "Original HPC provenance unresolved. " if name == 'earlier_report' else "Run-time config/commit snapshots unverified. "
    context += f"Exact biological match: {exact}; stored projection-aware correctness: {row['correct_selection']}."
    ui.render_case(row, report_context=context)


def main():
    st.set_page_config(page_title='Neuro-Symbolic Research Workbench', layout='wide')
    st.markdown(ui.PAGE_CSS, unsafe_allow_html=True)
    st.title('Neuro-Symbolic Research Workbench')
    st.caption('Synthetic bioprocess data · mechanistic evidence · auditable decisions')
    with st.sidebar:
        st.header('Explore the project')
        page = st.radio('View', ['Results', 'Synthetic data', 'Case explorer'], key='page')
        name = st.selectbox('Frozen study', list(RESULT_LABELS), format_func=RESULT_LABELS.get, key='study')
        st.caption('Replay of saved research results. No LLM server or trained models required.')
    if page == 'Synthetic data':
        render_samples()
    else:
        with st.spinner('Loading frozen evidence…'):
            rows = replay(name)
        if page == 'Results':
            render_results(name, rows)
        else:
            render_cases(name, rows)


if __name__ == '__main__':
    main()
