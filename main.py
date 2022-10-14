import argparse
import datetime
import json
import os
import random
import shutil 
from components import init_components, st_highlightable_text
import streamlit as st
from st_click_detector import click_detector
from annotated_text import annotated_text
import extra_streamlit_components as stx

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

from difflib import SequenceMatcher

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

EN_STOPWORDS = set(stopwords.words('english'))
FR_STOPWORDS = set(stopwords.words('french'))
CCFR_DEFAULT_VAL = -1
PREC_REC_DEFAULT_VAL = -1
CCFR_VALUES = (CCFR_DEFAULT_VAL, 0, 1, 2, 3, 4, 5)
PREC_VALUES = (PREC_REC_DEFAULT_VAL, 0, 25, 50, 75, 100)
REC_VALUES = (PREC_REC_DEFAULT_VAL, 0, 25, 50, 75, 100)


def _color_in_gold_sample(sent, gold_summary, stopwords):
    words_in_sent = sent.lower().split()
    gold_words = gold_summary.split()
    gold_words_to_color = []
    for w in gold_words:
        if w.lower() not in stopwords:
            if w.lower() in words_in_sent:
                gold_words_to_color.append(w)
            else:
                is_matched = False 
                for w_sent in words_in_sent:
                    if SequenceMatcher(a=w.lower(), b=w_sent).ratio() > 0.8:
                        is_matched = True 
                if is_matched:
                    gold_words_to_color.append(w)
    return gold_words_to_color


def _create_radios(model_name, doc_id, results_dir):
    def update_radio_coh():
        st.session_state[f'{model_name}_{doc_id}_coh_updated'] = st.session_state[f'{model_name}_{doc_id}_coh'] 
        with open(os.path.join(results_dir, f"{model_name}_{doc_id}_coh"), 'w') as fw:
            fw.write(str(st.session_state[f'{model_name}_{doc_id}_coh']) + "\n")

    def update_radio_con():
        st.session_state[f'{model_name}_{doc_id}_con_updated'] = st.session_state[f'{model_name}_{doc_id}_con'] 
        with open(os.path.join(results_dir, f"{model_name}_{doc_id}_con"), 'w') as fw:
            fw.write(str(st.session_state[f'{model_name}_{doc_id}_con']) + "\n")

    def update_radio_flu():
        st.session_state[f'{model_name}_{doc_id}_flu_updated'] = st.session_state[f'{model_name}_{doc_id}_flu'] 
        with open(os.path.join(results_dir, f"{model_name}_{doc_id}_flu"), 'w') as fw:
            fw.write(str(st.session_state[f'{model_name}_{doc_id}_flu']) + "\n")

    def update_radio_rel():
        st.session_state[f'{model_name}_{doc_id}_rel_updated'] = st.session_state[f'{model_name}_{doc_id}_rel'] 
        with open(os.path.join(results_dir, f"{model_name}_{doc_id}_rel"), 'w') as fw:
            fw.write(str(st.session_state[f'{model_name}_{doc_id}_rel']) + "\n")

    _, center, _ = st.columns([3, 8, 1])

    with center:
        st.radio(
            'Coherence', 
            CCFR_VALUES, 
            index=0 if f'{model_name}_{doc_id}_coh_updated' not in st.session_state else CCFR_VALUES.index(st.session_state[f'{model_name}_{doc_id}_coh_updated']),
            key=f'{model_name}_{doc_id}_coh', 
            on_change=update_radio_coh,
            horizontal=True,
        )
        st.radio(
            'Consistency',
            CCFR_VALUES, 
            index=0 if f'{model_name}_{doc_id}_con_updated' not in st.session_state else CCFR_VALUES.index(st.session_state[f'{model_name}_{doc_id}_con_updated']),
            key=f'{model_name}_{doc_id}_con', 
            on_change=update_radio_con,
            horizontal=True,
        )
        st.radio(
            'Fluency', 
            CCFR_VALUES, 
            index=0 if f'{model_name}_{doc_id}_flu_updated' not in st.session_state else CCFR_VALUES.index(st.session_state[f'{model_name}_{doc_id}_flu_updated']),
            key=f'{model_name}_{doc_id}_flu', 
            on_change=update_radio_flu,
            horizontal=True,
        )
        st.radio(
            'Relevance', 
            CCFR_VALUES, 
            index=0 if f'{model_name}_{doc_id}_rel_updated' not in st.session_state else CCFR_VALUES.index(st.session_state[f'{model_name}_{doc_id}_rel_updated']),
            key=f'{model_name}_{doc_id}_rel', 
            on_change=update_radio_rel,
            horizontal=True,
        )


def _display_placeholder_model(
    model_name, 
    subheader_str,
    gen_samples, 
    doc_id, 
    bigbird_n_sent, 
    layout_bigbird_n_sent,
    prec_results_dir,
    rec_results_dir,
    gold_samples,
    stopwords,
    placeholder_gold,
):
    with placeholder_gold.container():
        st.subheader("Ground-truth abstract")
        st.write(gold_samples[doc_id])
            
    text = gold_samples[doc_id].split()

    st.subheader(subheader_str)
    st.info("""
        Summary is split by sentence. 
        Click on any sentence to color the corresponding words in the ground-truth abstract. 
        Evaluate using per-sentence precision, recall, coherence, consistency, fluency, and relevance.
    """)
    
    def _update_gen_checkbox(sent_idx):
        for i in range(bigbird_n_sent):
            if model_name != "bigbird" or i != sent_idx:
                st.session_state[f"chk_bigbird_{doc_id}_{i}"] = False
        for i in range(layout_bigbird_n_sent):
            if model_name != "layout-bigbird" or i != sent_idx:
                st.session_state[f"chk_layout-bigbird_{doc_id}_{i}"] = False
        
    def _update_prec_eval(sent_idx):
        if f"{model_name}_{doc_id}_sent{sent_idx}_prec_updated" not in st.session_state:
            st.session_state[f'{model_name}_{doc_id}_sent{sent_idx}_prec_updated'] = st.session_state[f'{model_name}_{doc_id}_sent{sent_idx}_prec'] 
        st.session_state[f'{model_name}_{doc_id}_sent{sent_idx}_prec_updated'] = st.session_state[f'{model_name}_{doc_id}_sent{sent_idx}_prec'] 
        with open(os.path.join(prec_results_dir, f"{model_name}_{doc_id}_sent{sent_idx}_prec"), 'w') as fw:
            fw.write(str(st.session_state[f'{model_name}_{doc_id}_sent{sent_idx}_prec']) + "\n")

    def _highlight_and_color(sent_idx):
        colored_words = _color_in_gold_sample(
            gen_samples[doc_id][sent_idx],
            gold_samples[doc_id],
            stopwords
        )
        with placeholder_gold.container():
            st.subheader("Ground-truth abstract")
            if f"{doc_id}_last_checked" not in st.session_state:
                if f"{model_name}_{doc_id}_sent{sent_idx}_highlighted" in st.session_state:
                    text_to_highlight = st.session_state[f"{model_name}_{doc_id}_sent{sent_idx}_highlighted"]
                else:
                    text_to_highlight = [False for _ in range(len(text))]
            else:
                if st.session_state[f"{doc_id}_last_checked"] != model_name + "_" + str(sent_idx):
                    if f"{model_name}_{doc_id}_sent{sent_idx}_highlighted" not in st.session_state:
                        text_to_highlight = [False for _ in range(len(text))]
                    else:
                        text_to_highlight = st.session_state[f"{model_name}_{doc_id}_sent{sent_idx}_highlighted"]
                else:
                    text_to_highlight = [False for _ in range(len(text))]
            highlighted = st_highlightable_text(
                text, 
                text_to_highlight, 
                colored_words, 
                key=f"{model_name}_{doc_id}_{sent_idx}"
            )
            return highlighted

    updated = False
    for sent_idx, sent in enumerate(gen_samples[doc_id]):
        first_col, second_col, third_col = st.columns([1, 6, 7])
        with first_col:
            if sent_idx == 0:
                st.markdown("\n")
                st.markdown("\n")
            st.checkbox("", key=f'chk_{model_name}_{doc_id}_{sent_idx}', on_change=_update_gen_checkbox, args=(sent_idx,))
            # Checkbox has been checked
            if st.session_state[f"chk_{model_name}_{doc_id}_{sent_idx}"]:
                highlighted = _highlight_and_color(sent_idx)
                if highlighted is not None:
                    st.session_state[f"{model_name}_{doc_id}_sent{sent_idx}_highlighted"] = highlighted
                    _update_gen_gold_association(
                        highlighted, 
                        model_name, 
                        doc_id, 
                        text,
                        sent_idx,
                        gen_gold_results_dir
                    )
                    updated = True
                st.session_state[f"{doc_id}_last_checked"] = model_name + "_" + str(sent_idx)
        with second_col:
            if sent_idx == 0:
                st.markdown("**Sentence**")
            st.markdown(sent)
        with third_col:
            st.radio(
                "Precision (%)", 
                PREC_VALUES, 
                key=f'{model_name}_{doc_id}_sent{sent_idx}_prec', 
                on_change=_update_prec_eval, 
                args=(sent_idx,),
                index=0 if f'{model_name}_{doc_id}_sent{sent_idx}_prec_updated' not in st.session_state else PREC_VALUES.index(st.session_state[f'{model_name}_{doc_id}_sent{sent_idx}_prec_updated']),
                label_visibility="visible" if sent_idx == 0 else "collapsed",
                horizontal=True,
            )
    
    if updated:
        _update_rec_eval(
            model_name,
            doc_id,
            len(text),
            bigbird_n_sent if model_name == "bigbird" else layout_bigbird_n_sent,
            rec_results_dir
        )
    if f'{model_name}_{doc_id}_rec' in st.session_state:
        st.write(f"Recall = {round(st.session_state[f'{model_name}_{doc_id}_rec'] * 100, 2)} %")


def _update_rec_eval(
    model_name,
    doc_id,
    n_gold_words,
    n_sent,
    rec_results_dir,
):
    true_positive = 0
    for sent_idx in range(n_sent):
        if f"{model_name}_{doc_id}_sent{sent_idx}_highlighted" in st.session_state:
            sent_highlighted = st.session_state[f"{model_name}_{doc_id}_sent{sent_idx}_highlighted"]
            true_positive += sent_highlighted.count(True)
    recall = true_positive / n_gold_words
    st.session_state[f'{model_name}_{doc_id}_rec'] = recall
    rec_output_file = os.path.join(rec_results_dir, f"{model_name}_{doc_id}_rec")
    with open(rec_output_file, 'w') as fw:
        fw.write(str(recall) + "\n")

def _update_gen_gold_association(
    highlighted,
    model_name, 
    doc_id, 
    gold_words,
    current_sent_idx,
    gen_gold_results_dir,
):
    # Save current highlight
    sent_highlighted_in_doc = f"{model_name}_{doc_id}_sent{current_sent_idx}_highlighted"
    st.session_state[sent_highlighted_in_doc] = highlighted

    # Save highlighted words in gold sample
    gen_gold_assoc_file = os.path.join(gen_gold_results_dir, f"{model_name}_{doc_id}_sent{current_sent_idx}_matched")
    gold_matched = [gold_words[i] for i in range(len(gold_words)) if highlighted[i] == True]
    gold_matched = " ".join(gold_matched)
    with open(gen_gold_assoc_file, 'w') as fw:
        fw.write(" ".join([str(b) for b in highlighted]) + "\n")
        fw.write(gold_matched)
    

def _load_results_in_session_state(
    ccfr_results_dir, 
    prec_results_dir, 
    rec_results_dir, 
    gen_gold_assoc_dir,
    unable_to_eval_file
):
    ccfr_results_files = os.listdir(ccfr_results_dir)
    prec_results_files = os.listdir(prec_results_dir)
    rec_results_files = os.listdir(rec_results_dir)
    gen_gold_assoc_files = os.listdir(gen_gold_assoc_dir)

    for filename in ccfr_results_files:
        model_name, doc_id, cat = filename.split("_")
        with open(os.path.join(ccfr_results_dir, filename), 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        last_result = int(lines[-1])
        st.session_state[f'{model_name}_{doc_id}_{cat}_updated'] = last_result

    for filename in prec_results_files:
        model_name, doc_id, sent_idx, _ = filename.split("_")
        with open(os.path.join(prec_results_dir, filename), 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        last_result = int(lines[-1])
        st.session_state[f'{model_name}_{doc_id}_{sent_idx}_prec_updated'] = last_result

    for filename in rec_results_files:
        model_name, doc_id, _ = filename.split("_")
        with open(os.path.join(rec_results_dir, filename), 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        last_result = float(lines[-1])
        st.session_state[f'{model_name}_{doc_id}_rec'] = last_result

    for filename in gen_gold_assoc_files:
        model_name, doc_id, sent_idx, _ = filename.split("_")
        sent_idx = sent_idx.replace("sent", "")
        with open(os.path.join(gen_gold_assoc_dir, filename), 'r') as f:
            lines = f.readlines()
        highlight = lines[0].split()
        highlight = [True if h == "True" else False for h in highlight]
        st.session_state[f"{model_name}_{doc_id}_sent{sent_idx}_highlighted"] = highlight

    if os.path.isfile(unable_to_eval_file):
        with open(unable_to_eval_file, 'r') as f:
            unable_to_eval_doc_ids = f.readlines()
        unable_to_eval_doc_ids = [doc_id.strip() for doc_id in unable_to_eval_doc_ids]
        for doc_id in unable_to_eval_doc_ids:
            st.session_state[f'unable_to_eval_{doc_id}_checked'] = True 


def loralay_eval_interface(
    gold_samples, 
    bigbird_samples, 
    layout_bigbird_samples, 
    doc_id, 
    all_doc_ids,
    samples_lang,
    ccfr_results_dir,
    prec_results_dir,
    rec_results_dir,
    gen_gold_results_dir,
    unable_to_eval_file,
    models_A,
    samples_titles,
    samples_urls,
):
    st.title("LoRaLay Evaluation Interface")
    
    # Uncheck all radios
    st.markdown(
        """
            <style>div[role="radiogroup"] >  :first-child{
                display: none !important;
            }</style>
        """, 
        unsafe_allow_html=True
    )

    _load_results_in_session_state(
        ccfr_results_dir, 
        prec_results_dir, 
        rec_results_dir, 
        gen_gold_results_dir,
        unable_to_eval_file
    )

    last_idx = len(all_doc_ids) - 1

    if "doc_idx" not in st.session_state:
        st.session_state.doc_idx = all_doc_ids.index(doc_id)
    else:
        doc_id = all_doc_ids[st.session_state.doc_idx]
        with open("./last_doc_id.txt", 'w') as fw:
            fw.write(doc_id)

    if samples_lang[doc_id] == "en":
        stopwords = EN_STOPWORDS
    else:
        assert samples_lang[doc_id] == "fr"
        stopwords = FR_STOPWORDS

    st.header(f"Document {doc_id} ({st.session_state.doc_idx + 1}/{len(all_doc_ids)})")
    if doc_id in samples_titles:
        st.write(f"*{samples_titles[doc_id]}*")        
    if doc_id in samples_urls:
        st.write(f"[Link to article]({samples_urls[doc_id]})")

    
    chosen_id = stx.tab_bar(data=[
        stx.TabBarItemData(id="model_a", title="Model A", description=""),
        stx.TabBarItemData(id="model_b", title="Model B", description=""),
    ], default="model_a")

    bigbird_n_sent = len(bigbird_samples[doc_id])
    layout_bigbird_n_sent = len(layout_bigbird_samples[doc_id])

    if models_A[st.session_state.doc_idx] == "bigbird":
        model_A = "bigbird"
        model_A_samples = bigbird_samples
        model_B = "layout-bigbird"
        model_B_samples = layout_bigbird_samples
    else:
        model_A = "layout-bigbird"
        model_A_samples = layout_bigbird_samples
        model_B = "bigbird"
        model_B_samples = bigbird_samples

    placeholder_gold_a = st.empty()
    placeholder_gold_b = st.empty()

    if chosen_id == "model_a":
        _display_placeholder_model(
            model_A, 
            "Summary generated by model A",
            model_A_samples,
            doc_id, 
            bigbird_n_sent, 
            layout_bigbird_n_sent,
            prec_results_dir,
            rec_results_dir,
            gold_samples,
            stopwords,
            placeholder_gold_a,
        )
        _create_radios(model_A, doc_id, ccfr_results_dir)

    else:
        _display_placeholder_model(
            model_B, 
            "Summary generated by model B",
            model_B_samples, 
            doc_id, 
            bigbird_n_sent, 
            layout_bigbird_n_sent,
            prec_results_dir,
            rec_results_dir,
            gold_samples,
            stopwords,
            placeholder_gold_b,
        )
        
        _create_radios(model_B, doc_id, ccfr_results_dir)


    def _update_unable_to_eval(doc_id):
        if os.path.isfile(unable_to_eval_file):
            with open(unable_to_eval_file, 'r') as f:
                unable_to_eval_doc_ids = f.readlines()
            unable_to_eval_doc_ids = [doc_id.strip() for doc_id in unable_to_eval_doc_ids]
        else:
            unable_to_eval_doc_ids = []

        if (
            f"unable_to_eval_{doc_id}_checked" not in st.session_state
            or not st.session_state[f"unable_to_eval_{doc_id}_checked"]
        ):
            st.session_state[f"unable_to_eval_{doc_id}_checked"] = True 
            unable_to_eval_doc_ids.append(doc_id)
        else:
            st.session_state[f"unable_to_eval_{doc_id}_checked"] = False
            assert doc_id in unable_to_eval_doc_ids
            unable_to_eval_doc_ids.remove(doc_id)

        with open(unable_to_eval_file, 'w') as fw:
            for doc_id in unable_to_eval_doc_ids:
                fw.write(doc_id + "\n")       
     

    st.checkbox(
        "I am unable to evaluate this document.", 
        value=True if f'unable_to_eval_{doc_id}_checked' in st.session_state and st.session_state[f'unable_to_eval_{doc_id}_checked'] else False,
        key=f'chk_unable_to_eval_{doc_id}', 
        on_change=_update_unable_to_eval, 
        args=(doc_id,)
    )

    def _go_to_next():
        st.session_state.doc_idx += 1
    def _go_to_previous():
        st.session_state.doc_idx -= 1

    left, _, right = st.columns([1, 8, 1])
    with left:
        if st.session_state.doc_idx > 0:
            st.button('Previous', on_click=_go_to_previous)
    with right:
        if st.session_state.doc_idx < last_idx:
            next_is_disabled = False 
            for bigbird_sent_idx, layout_bigbird_sent_idx in zip(range(bigbird_n_sent), range(layout_bigbird_n_sent)):
                if f"bigbird_{doc_id}_sent{bigbird_sent_idx}_prec_updated" not in st.session_state:
                    next_is_disabled = True 
                    break
                if f"layout-bigbird_{doc_id}_sent{layout_bigbird_sent_idx}_prec_updated" not in st.session_state:
                    next_is_disabled = True 
                    break
            for cat in ["coh", "con", "flu", "rel"]:
                if f"bigbird_{doc_id}_{cat}_updated" not in st.session_state:
                    next_is_disabled = True 
                    break
                if f"layout-bigbird_{doc_id}_{cat}_updated" not in st.session_state:
                    next_is_disabled = True 
                    break
            
            # if f'unable_to_eval_{doc_id}_checked' in st.session_state and st.session_state[f'unable_to_eval_{doc_id}_checked']:
            if st.session_state[f'chk_unable_to_eval_{doc_id}']:
                next_is_disabled = False
            st.button('Next', on_click=_go_to_next, disabled=next_is_disabled)

        

def load_samples(samples, is_gold=False):
    samples = [json.loads(sample) for sample in samples]
    if is_gold:
        samples_lang = {
            sample["id"]: sample["lang"] for sample in samples
        }
        # samples_titles = {
        #     sample["id"]: sample["title"] for sample in samples
        # }
        samples_titles = dict()
        samples_urls = dict()
        for sample in samples: 
            if "title" in sample:
                samples_titles[sample["id"]] = sample["title"]
            if "pdf_url" in sample:
                samples_urls[sample["id"]] = sample["pdf_url"]
    else:
        samples_lang = None
        samples_titles = None
        samples_urls = None 
    samples = {
        sample["id"]: sample["abstract"].strip() if is_gold else sent_tokenize(sample["output"].strip()) for sample in samples
    }
    return samples, (samples_lang, samples_titles, samples_urls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--path_to_gold",
        type=str,
        default="samples/gold.txt",
    )
    parser.add_argument(
        "--path_to_bigbird",
        type=str,
        default="samples/bigbird.txt",
    )
    parser.add_argument(
        "--path_to_layout_bigbird",
        type=str,
        default="samples/layout-bigbird.txt",
    )
    parser.add_argument(
        "--path_to_models_A",
        type=str,
        default="samples/models_A.txt",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="eval_results",
    )
    parser.add_argument(
        "--overwrite_eval",
        action="store_true", 
    )
    parser.add_argument(
        "--dev",
        action="store_true", 
        help="Runs in development mode (frontend)"
    )
    args = parser.parse_args()


    init_components(args.dev)

    with open(args.path_to_gold) as f:
        gold_samples = f.readlines()
    with open(args.path_to_bigbird) as f:
        bigbird_samples = f.readlines()
    with open(args.path_to_layout_bigbird) as f:
        layout_bigbird_samples = f.readlines()
    with open(args.path_to_models_A) as f:
        models_A = f.readlines()

    gold_samples, (samples_lang, samples_titles, samples_urls) = load_samples(gold_samples, is_gold=True)
    bigbird_samples, _ = load_samples(bigbird_samples)
    layout_bigbird_samples,  _ = load_samples(layout_bigbird_samples)
    models_A = [model.strip() for model in models_A]

    all_doc_ids = tuple([sample_id for sample_id, _ in gold_samples.items()])
    # all_doc_ids = sorted(all_doc_ids)

    ccfr_results_dir = os.path.join(args.results_dir, "ccfr_outputs")
    prec_results_dir = os.path.join(args.results_dir, "precision_outputs")
    rec_results_dir = os.path.join(args.results_dir, "recall_outputs")
    gen_gold_results_dir = os.path.join(args.results_dir, "gen_gold_assoc")
    unable_to_eval_file = os.path.join(args.results_dir, "unable_to_eval.txt")

    @st.cache
    def prepare_results_dir():
        if os.path.isdir(args.results_dir):
            if args.overwrite_eval:
               
                if os.path.isdir(ccfr_results_dir):
                    shutil.rmtree(ccfr_results_dir)
                if os.path.isdir(prec_results_dir):
                    shutil.rmtree(prec_results_dir)
                if os.path.isdir(rec_results_dir):
                    shutil.rmtree(rec_results_dir)
                if os.path.isdir(gen_gold_results_dir):
                    shutil.rmtree(gen_gold_results_dir)
                
                os.makedirs(ccfr_results_dir)
                os.makedirs(prec_results_dir)
                os.makedirs(rec_results_dir)
                os.makedirs(gen_gold_results_dir)
                if os.path.isfile("./last_doc_id.txt"):
                    os.remove("./last_doc_id.txt")
                if os.path.isfile(unable_to_eval_file):
                    os.remove(unable_to_eval_file)
            else:
                if not os.path.isdir(ccfr_results_dir):
                    os.makedirs(ccfr_results_dir)
                if not os.path.isdir(prec_results_dir):
                    os.makedirs(prec_results_dir)
                if not os.path.isdir(rec_results_dir):
                    os.makedirs(rec_results_dir)
                if not os.path.isdir(gen_gold_results_dir):
                    os.makedirs(gen_gold_results_dir)
        else:
            os.mkdir(args.results_dir)
            os.makedirs(ccfr_results_dir)
            os.makedirs(prec_results_dir)
            os.makedirs(rec_results_dir)
            os.makedirs(gen_gold_results_dir)

    prepare_results_dir()

    if os.path.isfile("./last_doc_id.txt"):
        with open("./last_doc_id.txt", 'r') as f:
            doc_id = f.read().strip()
    else:
        doc_id = all_doc_ids[0]

    loralay_eval_interface(
        gold_samples, 
        bigbird_samples, 
        layout_bigbird_samples, 
        doc_id, 
        all_doc_ids,
        samples_lang,
        ccfr_results_dir,
        prec_results_dir,
        rec_results_dir,
        gen_gold_results_dir,
        unable_to_eval_file,
        models_A,
        samples_titles,
        samples_urls
    )
