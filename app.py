# -*- coding: utf-8 -*-
"""
通知表所見ドラフト作成アプリ（Streamlit + Claude API）
実行: pip install streamlit anthropic && streamlit run app.py
"""
import hmac
import json
import re
import unicodedata
from datetime import datetime
import pandas as pd
import streamlit as st

try:
    import anthropic
except ImportError:  # 未インストール時の案内
    anthropic = None

try:  # ブラウザ自動保存（localStorage）用。未インストールでもアプリ自体は動作します
    from streamlit_js_eval import streamlit_js_eval
except ImportError:
    streamlit_js_eval = None

LS_KEY = "shoken_draft_v1"

st.set_page_config(page_title="通知表 所見ドラフト作成", page_icon="📝", layout="centered")

DEFAULT_MODEL = "claude-sonnet-5-5"
TOLERANCE = 10

TONES = {
    "① 温かみ・共感重視": "思いやり、誠実さを大切にし、情緒的でやわらかな表現を用いる。子どもの気持ちに寄り添う温かな語り口にする。",
    "② 成長・チャレンジ重視": "主体性、変容、挑戦するエネルギーを前面に出す。『〜に挑戦し』『〜と成長しました』など伸びていく姿を力強く価値付ける。",
    "③ 落ち着き・思考重視": "深い思考、丁寧さ、着実な追究の姿勢を中心に描く。落ち着いた筆致で、考えを深める過程を丁寧に表現する。",
}

TAGS = {
    "生活面": ["あいさつが丁寧", "時間を守る", "整理整頓が上手", "最後まで責任を持つ",
              "思いやりがある", "友達に優しい", "正義感が強い", "素直で明るい", "気配り上手", "約束を守る"],
    "学習面": ["意欲的に発言する", "粘り強く取り組む", "ノートが丁寧", "考えを筋道立てて説明できる",
              "友達の意見を聞いて考えを深める", "自分から課題を見つける", "計算が正確", "読書が好き",
              "表現が豊か", "振り返りができる", "進んで質問する", "見通しをもって取り組む"],
    "係・委員会・行事": ["係活動に熱心", "リーダーシップがある", "縁の下の力持ち", "クラスを盛り上げる",
                      "行事で力を発揮", "当番活動を忘れず行う", "下級生にやさしい", "協力して活動を進める"],
}

SYSTEM_PROMPT = """あなたは経験豊富な小学校のベテラン教員です。通知表（通信表）の「所見」の下書きを作成します。

【文章の方針】
- 小学校学習指導要領（新学習指導要領）の三つの柱を意識して、子どもの姿を価値付けること。
  1. 知識及び技能の習得（確かな知識・技能が身に付いている姿）
  2. 思考力・判断力・表現力等の育成（考え、判断し、表現する姿）
  3. 学びに向かう力・人間性等（主体的・対話的で深い学びの姿勢、粘り強さ、自己調整）
  ※入力された情報に関係する観点を中心に自然に織り込み、観点を無理に全て網羅したり、観点名をそのまま列挙したりしないこと。
- ベテラン教員らしく、具体的な場面に基づく説得力と、保護者に伝わる温かみのある文体にすること。
- 文体は「です・ます調」。「〜さん」などの呼称や児童名・イニシャル・出席番号は本文に書かないこと（主語は省略する）。
- 否定的な表現や課題は、「〜することでさらに伸びていくでしょう」のように前向きな価値付けに必ず変換すること。
- 入力にない事実（行事名、成績、エピソード）を創作しないこと。入力が少ない場合は一般的な表現にとどめること。
- 指定された文字数（目標±10字以内）を厳守すること。改行は入れず一段落で書くこと。
- 出力は所見本文のみ。挨拶、前置き、見出し、説明、括弧書きの文字数表記は一切出力しないこと。"""


# ---------------------------------------------------------------- helpers
def count_chars(text: str) -> int:
    return len(re.sub(r"[\r\n]", "", text or ""))


def clean_output(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^[「『]|[」』]$", "", t)
    return re.sub(r"[\r\n]+", "", t).strip()


def call_claude(api_key: str, model: str, user_prompt: str) -> str:
    if anthropic is None:
        raise RuntimeError("anthropic ライブラリが未インストールです。`pip install anthropic` を実行してください。")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = clean_output("".join(b.text for b in resp.content if getattr(b, "type", "") == "text"))
    if not text:
        raise RuntimeError(
            f"AIの応答が空でした（stop_reason={getattr(resp, 'stop_reason', '不明')}）。"
            "もう一度お試しいただくか、詳細設定のモデル名をご確認ください。")
    return text


def friendly_error(e: Exception) -> str:
    if anthropic is not None:
        if isinstance(e, anthropic.AuthenticationError):
            return "APIキーが正しくないようです。サイドバーのキーをご確認ください。"
        if isinstance(e, anthropic.RateLimitError):
            return "リクエストが集中しています。少し待ってからもう一度お試しください。"
        if isinstance(e, anthropic.APIConnectionError):
            return "通信に失敗しました。ネットワーク環境をご確認ください。"
        if isinstance(e, anthropic.NotFoundError):
            return "モデル名が見つかりません。サイドバーの詳細設定でモデル名をご確認ください。"
        if isinstance(e, anthropic.APIStatusError):
            return f"APIエラーが発生しました（{e.status_code}）。"
    return f"エラーが発生しました：{e}"


def build_generate_prompt(term, tone_name, target, tags, memo):
    return f"""次の情報をもとに、{term}の通知表の所見を作成してください。

【目標文字数】{target}字（{target - TOLERANCE}〜{target + TOLERANCE}字に収めること）
【文章のトーン】{tone_name}：{TONES[tone_name]}
【長所・行動の特徴（選択タグ）】{('、'.join(tags)) if tags else '（なし）'}
【教員メモ】
{memo.strip() if memo.strip() else '（なし）'}

所見本文のみを出力してください。"""


def build_adjust_prompt(text, target, delta_hint):
    now = count_chars(text)
    return f"""次の所見を、内容と文体を保ったまま{delta_hint}。
現在{now}字です。調整後は{target - TOLERANCE}〜{target + TOLERANCE}字（目標{target}字）に収めてください。
事実の追加・創作はせず、前向きな表現は維持してください。所見本文のみを出力してください。

【現在の所見】
{text}"""


def generate_with_fix(api_key, model, prompt, target, max_fix=2):
    text = call_claude(api_key, model, prompt)
    for _ in range(max_fix):
        diff = count_chars(text) - target
        if abs(diff) <= TOLERANCE:
            break
        hint = (f"{abs(diff)}字ほど削って短くしてください" if diff > 0
                else f"{abs(diff)}字ほど加えて充実させてください")
        try:
            text = call_claude(api_key, model, build_adjust_prompt(text, target, hint))
        except RuntimeError:  # 調整の応答が空だった場合は、調整前の所見を残す
            break
    return text


def rec_key(term, ident):
    return f"{term}::{ident}"


def sort_ident(ident):
    return (0, int(ident)) if ident.isdigit() else (1, ident)


# ---------------------------------------------------------------- state
st.session_state.setdefault("records", {})  # {"1学期::3": {"term","ident","text","tone"}}
st.session_state.setdefault("flash", None)
st.session_state.setdefault("ls_loaded", False)
st.session_state.setdefault("ls_n", 0)


def serialize_records() -> str:
    return json.dumps({"records": st.session_state["records"]}, ensure_ascii=False, sort_keys=True)


def autoload():
    """起動時にブラウザの自動保存データを読み込む（表示前に実行）。"""
    ss = st.session_state
    if streamlit_js_eval is None or ss["ls_loaded"]:
        return
    res = streamlit_js_eval(
        js_expressions=f"JSON.stringify({{v: localStorage.getItem('{LS_KEY}')}})", key="ls_read")
    if res is None:  # ブラウザからの応答待ち（この間は保存しない）
        return
    try:
        wrapper = json.loads(res) if isinstance(res, str) else res
        raw = wrapper.get("v") if isinstance(wrapper, dict) else None
        if raw:
            for k, r in json.loads(raw).get("records", {}).items():
                if k in ss["records"] or not all(x in r for x in ("term", "ident", "text")):
                    continue
                ss["records"][k] = {"term": r["term"], "ident": str(r["ident"]),
                                    "text": r["text"], "tone": r.get("tone", "")}
                ss[f"edit::{k}"] = r["text"]
            if ss["records"]:
                ss["flash"] = ("success", f"自動保存されたデータ（{len(ss['records'])}件）を復元しました。")
    except Exception:  # noqa: BLE001
        pass
    ss["ls_loaded"] = True
    ss["ls_last"] = serialize_records()


def autosave():
    """変更があればブラウザ（localStorage）へ保存する（画面の最後に実行）。"""
    ss = st.session_state
    if streamlit_js_eval is None or not ss["ls_loaded"] or not ss.get("autosave", True):
        return
    payload = serialize_records()
    if payload != ss.get("ls_last"):
        ss["ls_last"] = payload
        ss["ls_n"] += 1
        ss["ls_expr"] = f"localStorage.setItem('{LS_KEY}', {json.dumps(payload)})"
        ss["ls_wkey"] = f"ls_write_{ss['ls_n']}"
    if ss.get("ls_expr"):  # 同じkey・同じ内容の再描画では再実行されない
        streamlit_js_eval(js_expressions=ss["ls_expr"], key=ss["ls_wkey"])


def get_secret(name: str) -> str:
    """Streamlit の Secrets から値を取得（未設定なら空文字）。"""
    try:
        return str(st.secrets[name]).strip()
    except Exception:  # noqa: BLE001
        return ""


def resolve_api_key() -> str:
    """Secrets に登録されたキーを優先し、なければサイドバーの入力値を使う。"""
    return get_secret("ANTHROPIC_API_KEY") or st.session_state.get("api_key", "").strip()


# 任意：Secrets に APP_PASSWORD を設定すると、アプリ起動時にパスワードを求めます
_app_pw = get_secret("APP_PASSWORD")
if _app_pw and not st.session_state.get("authed"):
    st.title("📝 通知表 所見ドラフト作成")
    with st.form("login"):
        _pw = st.text_input("パスワード", type="password")
        if st.form_submit_button("入る"):
            if hmac.compare_digest(_pw.encode("utf-8"), _app_pw.encode("utf-8")):
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("パスワードが違います。")
    st.stop()

autoload()


def get_ident():
    return st.session_state.get("cur_ident", "")


def current_key():
    return rec_key(st.session_state["term_sel"], get_ident())


def set_text(key, text, term, ident, tone):
    st.session_state["records"][key] = {"term": term, "ident": ident, "text": text, "tone": tone}
    st.session_state[f"edit::{key}"] = text  # 次の描画で編集欄に反映


def cb_generate():
    ss = st.session_state
    api_key, ident = resolve_api_key(), get_ident()
    if not api_key:
        ss["flash"] = ("error", "サイドバーにAnthropic APIキーを入力してください。")
        return
    if not ident:
        ss["flash"] = ("error", "識別子（出席番号またはイニシャル）を入力してください。")
        return
    if not ss.get("tags_sel") and not ss.get("memo_sel", "").strip():
        ss["flash"] = ("error", "タグか3行メモのどちらかは入力してください。")
        return
    try:
        prompt = build_generate_prompt(ss["term_sel"], ss["tone_sel"], ss["target_sel"],
                                       ss.get("tags_sel", []), ss.get("memo_sel", ""))
        text = generate_with_fix(api_key, ss["model_name"], prompt, ss["target_sel"])
        set_text(current_key(), text, ss["term_sel"], ident, ss["tone_sel"])
        ss["flash"] = ("success", "所見のドラフトを作成しました。必要に応じて下で編集してください。")
    except Exception as e:  # noqa: BLE001
        ss["flash"] = ("error", friendly_error(e))


def cb_adjust(direction):
    ss = st.session_state
    key = current_key()
    api_key = resolve_api_key()
    rec = ss["records"].get(key)
    if not api_key:
        ss["flash"] = ("error", "サイドバーにAnthropic APIキーを入力してください。")
        return
    text = ss.get(f"edit::{key}", rec["text"] if rec else "")
    if not text.strip():
        ss["flash"] = ("error", "調整する所見がありません。先に生成または入力してください。")
        return
    hint = "10〜15字ほど削って短くしてください" if direction == "cut" else "10〜15字ほど加えて少し充実させてください"
    try:
        new = call_claude(api_key, ss["model_name"], build_adjust_prompt(text, ss["target_sel"], hint))
        set_text(key, new, ss["term_sel"], get_ident(), ss["tone_sel"])
        ss["flash"] = ("success", "文字数を調整しました。")
    except Exception as e:  # noqa: BLE001
        ss["flash"] = ("error", friendly_error(e))


def backup_json() -> bytes:
    data = {"app": "shoken-draft", "version": 1, "records": st.session_state["records"]}
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def cb_restore():
    ss = st.session_state
    f = ss.get("restore_file")
    if f is None:
        ss["flash"] = ("error", "復元するバックアップファイル（.json）を選んでください。")
        return
    apply_backup(f.getvalue().decode("utf-8-sig", errors="replace"))


def cb_restore_text():
    ss = st.session_state
    raw = ss.get("restore_text", "").strip()
    if not raw:
        ss["flash"] = ("error", "貼り付け欄にバックアップのテキストを入力してください。")
        return
    apply_backup(raw)


def apply_backup(raw: str):
    ss = st.session_state
    try:
        data = json.loads(raw)
        recs = data["records"]
        n = 0
        for k, r in recs.items():
            if not all(x in r for x in ("term", "ident", "text")):
                continue
            ss["records"][k] = {"term": r["term"], "ident": str(r["ident"]),
                                "text": r["text"], "tone": r.get("tone", "")}
            ss[f"edit::{k}"] = r["text"]
            n += 1
        ss["flash"] = ("success", f"{n}件の所見を復元しました。")
    except Exception:  # noqa: BLE001
        ss["flash"] = ("error", "バックアップファイルを読み込めませんでした。このアプリで書き出した .json か確認してください。")


def cb_delete():
    key = current_key()
    st.session_state["records"].pop(key, None)
    st.session_state.pop(f"edit::{key}", None)
    st.session_state["flash"] = ("success", "この児童の所見を削除しました。")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ 設定")
    if get_secret("ANTHROPIC_API_KEY"):
        st.success("🔑 APIキーは Secrets から読み込み済みです。入力は不要です。")
    else:
        st.text_input(
            "Anthropic API Key", type="password", key="api_key", placeholder="sk-ant-...",
            help="console.anthropic.com にログインし、「API Keys」で作成したキーをここに貼り付けます。"
                 "キーはこの画面のセッション内でのみ使われ、ファイルには保存されません。"
                 "（Secrets に ANTHROPIC_API_KEY を登録すると、入力が不要になります）",
        )
    ident_mode = st.radio("児童の識別方法", ["出席番号 1〜50番", "イニシャル入力"], key="ident_mode")
    term = st.radio("対象学期", ["1学期", "2学期", "3学期"], horizontal=True, key="term_sel")
    st.slider("目標文字数", 80, 200, 120, step=5, key="target_sel")
    st.radio("文章のトーン", list(TONES.keys()), key="tone_sel")
    with st.expander("詳細設定"):
        st.text_input("モデル名", value=DEFAULT_MODEL, key="model_name")
    st.caption("🔒 児童の実名は入力しないでください。出席番号かイニシャルのみをご利用ください。")

    st.divider()
    st.subheader("💾 保存とバックアップ")
    if streamlit_js_eval is None:
        st.warning("自動保存には `pip install streamlit-js-eval` が必要です（requirements.txt に追記してください）。")
    else:
        st.toggle("このiPad/ブラウザへ自動保存", value=True, key="autosave",
                  help="所見の変更をこの端末のブラウザ内に自動で保存し、次回開いたときに復元します。"
                       "APIキーは保存しません。共用の端末では、使用後に所見を削除するか、オフにしてください。")
        st.caption("※ 閲覧履歴やサイトデータを消去すると自動保存も消えます。大切なデータは下で書き出しておくと安心です。")
    n_all = len(st.session_state["records"])
    st.download_button(
        f"バックアップを書き出す（{n_all}件）", data=b"\xef\xbb\xbf" + backup_json(),
        file_name=f"shoken_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain", use_container_width=True, disabled=n_all == 0,
        help="保存先に「Googleドライブ」を選べます。APIキーは含まれません。",
    )
    st.file_uploader("バックアップを読み込む（.txt）", type=["txt", "json"], key="restore_file")
    st.button("📂 読み込んで復元", on_click=cb_restore, use_container_width=True)

    with st.expander("ファイルが使えない場合：テキストでコピー＆貼り付け"):
        st.caption("下のコードの右上のコピーボタンでコピーし、メモ等に貼って保管できます。復元するときは貼り付け欄に貼ってください。")
        st.code(backup_json().decode("utf-8"), language="json")
        st.text_area("復元用の貼り付け欄", key="restore_text", height=100)
        st.button("📋 貼り付けた内容から復元", on_click=cb_restore_text, use_container_width=True)

# ---------------------------------------------------------------- main
st.title("📝 通知表 所見ドラフト作成")
st.caption("タグと3行メモから、学習指導要領の観点を踏まえた所見の下書きを作ります。")

if anthropic is None:
    st.error("anthropic ライブラリが見つかりません。`pip install anthropic` を実行して再起動してください。")
if not resolve_api_key():
    st.info("👈 まずサイドバーで Anthropic API Key を入力してください（iPadでは左上の「≫」から開けます）。")

def normalize_ident(raw, mode):
    s = unicodedata.normalize("NFKC", "" if pd.isna(raw) else str(raw)).strip()
    if mode.startswith("出席番号"):
        return str(int(s)) if s.isdigit() and 1 <= int(s) <= 50 else None
    return s if re.fullmatch(r"[A-Za-z0-9.\-_・]{1,8}", s) else None


def run_batch(df, api_key):
    ss = st.session_state
    term_, tone_, target_ = ss["term_sel"], ss["tone_sel"], ss["target_sel"]
    jobs, problems, seen = [], [], set()
    for i, row in df.iterrows():
        raw_id, memo = row.get("識別子"), row.get("特徴・メモ")
        memo = "" if pd.isna(memo) else str(memo).strip()
        if (pd.isna(raw_id) or not str(raw_id).strip()) and not memo:
            continue  # 空行は無視
        ident_ = normalize_ident(raw_id, ident_mode)
        if ident_ is None:
            problems.append(f"{i + 1}行目：識別子「{'' if pd.isna(raw_id) else raw_id}」が正しくありません。")
        elif not memo:
            problems.append(f"{i + 1}行目（{ident_}）：特徴・メモが空のためスキップしました。")
        elif ident_ in seen:
            problems.append(f"{i + 1}行目（{ident_}）：識別子が重複しているためスキップしました。")
        elif ss.get("batch_skip", True) and rec_key(term_, ident_) in ss["records"]:
            problems.append(f"{ident_}：すでに所見があるためスキップしました。")
        else:
            seen.add(ident_)
            jobs.append((ident_, memo))
    if not jobs:
        st.warning("作成できる行がありませんでした。")
    else:
        bar = st.progress(0.0, text="作成を始めます…")
        done, errors = 0, []
        for n, (ident_, memo) in enumerate(jobs):
            bar.progress(n / len(jobs), text=f"{ident_} を作成中…（{n + 1}/{len(jobs)}）")
            try:
                prompt = build_generate_prompt(term_, tone_, target_, [], memo)
                text = generate_with_fix(api_key, ss["model_name"], prompt, target_)
                set_text(rec_key(term_, ident_), text, term_, ident_, tone_)
                done += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"{ident_}：{friendly_error(e)}")
                if anthropic is not None and isinstance(e, anthropic.AuthenticationError):
                    break
        bar.progress(1.0, text="完了")
        if done:
            st.success(f"{term_}の所見を{done}名分作成しました。下の「④ 一括出力」で確認できます。")
        for m in errors:
            st.error(m)
    for m in problems:
        st.warning(m)


with st.expander("👥 まとめて作成（複数人を一度に）", expanded=False):
    st.caption("1人1行で「識別子：特徴・メモ」の形に入力し、まとめて生成します。サイドバーの学期・文字数・トーンが全員に適用されます。"
               "生成中は画面を閉じず、そのままお待ちください。")
    batch_text = st.text_area(
        "1人1行で入力（識別子＋コロン＋メモ）", key="batch_text", height=220,
        placeholder="1：係活動に熱心。周りの人が嫌がる仕事も進んで引き受ける\n"
                    "2：算数で友達の考えを聞き、自分の式を直す。発表が増えた\n"
                    "3：誰にでも優しく、班の話し合いを盛り上げる",
        help="出席番号（またはイニシャル）のあとに「：」「:」「、」または空白を入れて、特徴・メモを書いてください。",
    )
    st.checkbox("すでに所見がある児童は上書きしない", value=True, key="batch_skip")
    if st.button("✨ まとめて生成する", use_container_width=True, key="batch_go"):
        _k = resolve_api_key()
        if not _k:
            st.error("サイドバーにAnthropic APIキーを入力してください。")
        else:
            _rows = []
            for _line in batch_text.splitlines():
                if not _line.strip():
                    continue
                _m = re.match(r"^\s*([A-Za-z0-9０-９.\-_・]+)\s*[:：,，、\s]\s*(.+)$", _line)
                _rows.append((_m.group(1), _m.group(2)) if _m else ("", _line.strip()))
            run_batch(pd.DataFrame(_rows, columns=["識別子", "特徴・メモ"]), _k)

st.subheader("① 児童を選ぶ")
if ident_mode == "出席番号 1〜50番":
    num = st.selectbox("出席番号", list(range(1, 51)), key="num_sel")
    st.session_state["cur_ident"] = str(num)
else:
    ini = st.text_input("イニシャル（例：T.K）", max_chars=8, key="ini_sel").strip()
    if len(ini) > 0 and not re.fullmatch(r"[A-Za-z0-9.\-_・]+", ini):
        st.warning("実名の入力を避けるため、イニシャルは英数字とピリオドでご入力ください。")
        ini = ""
    st.session_state["cur_ident"] = ini

ident = get_ident()
key = current_key() if ident else None
if ident and key in st.session_state["records"]:
    st.success(f"✅ {term}・{ident} の所見は保存済みです（下で再編集できます）。")

st.subheader("② 長所・行動の特徴")
all_tags = [t for group in TAGS.values() for t in group]
st.multiselect("当てはまるものを選択（複数可）", all_tags, key="tags_sel",
               help="生活面・学習面・係活動などから選べます。")

st.text_area(
    "3行メモ",
    key="memo_sel", height=150,
    placeholder="【場面・行動】例）算数の授業で、友達の考えを聞いて自分の式を直していた\n"
                "【内面・変容】例）間違いを恐れず発表できるようになった\n"
                "【周囲への影響】例）班の話し合いが活発になった",
    help="箇条書きでOKです。具体的な場面があるほど説得力のある所見になります。",
)

st.button("✨ 所見ドラフトを生成する", type="primary", use_container_width=True,
          on_click=cb_generate, disabled=not ident)

flash = st.session_state.get("flash")
if flash:
    (st.success if flash[0] == "success" else st.error)(flash[1])
    st.session_state["flash"] = None

# ---- preview
if ident and key in st.session_state["records"]:
    st.subheader("③ プレビュー・微調整")
    rec = st.session_state["records"][key]
    ekey = f"edit::{key}"
    if ekey not in st.session_state:
        st.session_state[ekey] = rec["text"]
    edited = st.text_area("所見（自由に編集できます）", key=ekey, height=200)
    rec["text"] = edited

    target = st.session_state["target_sel"]
    n = count_chars(edited)
    diff = n - target
    ok = abs(diff) <= TOLERANCE
    color = "#1a9a3a" if ok else "#d9480f"
    sign = "+" if diff > 0 else ""
    st.markdown(
        f"<div style='font-size:1.3rem;font-weight:bold;color:{color};'>"
        f"{n}字 <span style='font-size:0.95rem;'>（目標{target}字 / 差 {sign}{diff}字）"
        f"{' ✅ 範囲内' if ok else ' ⚠️ 範囲外'}</span></div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.button("✂️ あと10〜15字削る", use_container_width=True, on_click=cb_adjust, args=("cut",))
    c2.button("➕ あと10〜15字増やす", use_container_width=True, on_click=cb_adjust, args=("add",))
    st.button("🗑️ この児童の所見を削除", on_click=cb_delete)

# ---------------------------------------------------------------- bulk output
st.divider()
st.subheader("④ 一括出力")
view_term = st.radio("出力する学期", ["1学期", "2学期", "3学期"], horizontal=True,
                     index=["1学期", "2学期", "3学期"].index(term), key="view_term")
rows = sorted(
    [r for r in st.session_state["records"].values() if r["term"] == view_term and r["text"].strip()],
    key=lambda r: sort_ident(r["ident"]),
)

if not rows:
    st.info(f"{view_term}の所見はまだありません。")
else:
    st.markdown(f"**{view_term}：{len(rows)}名分**")
    with st.expander("一覧を見る", expanded=False):
        for r in rows:
            st.markdown(f"**{r['ident']}**（{count_chars(r['text'])}字）  \n{r['text']}")

    with_header = st.checkbox("1行目に見出し（識別子／所見／文字数）を付ける", value=False)
    lines = [f"{r['ident']}\t{re.sub(r'[\t\r\n]', '', r['text'])}\t{count_chars(r['text'])}" for r in rows]
    if with_header:
        lines.insert(0, "識別子\t所見本文\t文字数")
    tsv = "\n".join(lines)

    st.markdown("**📋 Googleスプレッドシートコピペ用エリア**")
    st.caption("右上のコピーボタンをタップ → スプレッドシートのA1セルを選んで貼り付けると、3列にきれいに分かれます。")
    st.code(tsv, language=None)
    st.text_area("（予備）長押しで全選択してコピーする場合はこちら", value=tsv, height=140, key=f"tsv_area_{view_term}")

    txt_lines = [f"【{view_term}】"] + [f"{r['ident']}\t{r['text']}\t{count_chars(r['text'])}字" for r in rows]
    st.download_button(
        "💾 .txt 一括ダウンロード", data=b"\xef\xbb\xbf" + "\n".join(txt_lines).encode("utf-8"),
        file_name=f"shoken_term{view_term[0]}.txt", mime="text/plain", use_container_width=True,
        help="保存先に「Googleドライブ」を選べます。",
    )

st.caption("※ 生成された文章は下書きです。最終的な内容は必ず先生ご自身でご確認ください。"
           "所見はこのブラウザに自動保存されますが、念のためサイドバーのバックアップも併用してください。")

autosave()
