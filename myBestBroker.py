import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ====================================================
# 🎯 CONFIGURAZIONE APP
# ====================================================

st.set_page_config(
    page_title="MyBestBroker - Confronta le commissioni di trading",
    layout="wide"
)

# ====================================================
# 🏦 TITOLO E DESCRIZIONE
# ====================================================

st.title("💹 MyBestBroker")
st.markdown("""
### Confronta le commissioni di trading dei tuoi broker

**MyBestBroker** è uno strumento di **confronto** che ti permette di analizzare le **commissioni di trading** applicate da diversi broker o banche in base a: importo dell’operazione, asset class, struttura delle commissioni (minimo, percentuale, massimo) e operatività stimata su base annua.

#### 🔍 Istruzioni per l'uso
1. Inserisci l’**importo medio** delle tue operazioni  
2. Seleziona un’**asset class**  
3. Configura uno o più **broker/banca** (puoi aggiungerli o rimuoverli)  
4. Inserisci le **commissioni** per ciascun broker  
5. Analizza:
   - 🥇 il confronto sul singolo trade  
   - 📊 le fasce di convenienza  
   - 📈 l’andamento dei costi  
   - 📅 il **Consigliatore annuale** per stimare il costo totale annuo        

""")

# ====================================================
# 📊 ASSET CLASS
# ====================================================

ASSET_CLASSES = [
    "Azioni Italia",
    "Azioni Europa",
    "Azioni USA",
    "Obbligazioni Italia"
]

# ====================================================
# 🧠 INIZIALIZZAZIONE SESSION STATE
# ====================================================

if "brokers_config" not in st.session_state:
    st.session_state.brokers_config = {
        "Broker 1": {
            "Azioni Italia": {"min": 6.0, "pct": 0.1, "max": 30},
            "Azioni Europa": {"min": 6.0, "pct": 0.15, "max": None},
            "Azioni USA": {"min": 6.0, "pct": 0.15, "max": None},
            "Obbligazioni Italia": {"min": 6.0, "pct": 0.1, "max": None},
        },
        "Broker 2": {
            "Azioni Italia": {"min": 0.5, "pct": 0.165, "max": 13.5},
            "Azioni Europa": {"min": 5.0, "pct": 0.18, "max": 13.5},
            "Azioni USA": {"min": 5.0, "pct": 0.18, "max": 13.5},
            "Obbligazioni Italia": {"min": 0.5, "pct": 0.09, "max": None},
        },
        "Broker 3": {
            "Azioni Italia": {"min": 2.95, "pct": 0.19, "max": 19},
            "Azioni Europa": {"min": 2.95, "pct": 0.19, "max": 19},
            "Azioni USA": {"min": 3.95, "pct": 0.19, "max": 19},
            "Obbligazioni Italia": {"min": 2.95, "pct": 0.19, "max": 19},
        },
    }

# ====================================================
# 🔧 SIDEBAR – INPUT PRINCIPALI
# ====================================================
# st.sidebar.markdown("""
# # 🔍 Istruzioni per l'uso
# 1. Inserisci l’**importo medio** delle tue operazioni  
# 2. Seleziona un’**asset class**  
# 3. Configura uno o più **broker/banca** (puoi aggiungerli o rimuoverli)  
# 4. Inserisci le **commissioni** per ciascun broker  
# 5. Analizza:
#    - 🥇 il confronto sul singolo trade  
#    - 📊 le fasce di convenienza  
#    - 📈 l’andamento dei costi  
#    - 📅 il **Consigliatore annuale** per stimare il costo totale annuo
# """)
# st.sidebar.markdown("---")
st.sidebar.header("💶 Importo medio delle tue transazioni (€)")
importo_medio = st.sidebar.number_input(
    "Inserisci l’importo medio",
    value=2000,
    min_value=10
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Impostazioni commissioni")

asset_class = st.sidebar.selectbox("Asset class", ASSET_CLASSES)

# ====================================================
# 🏦 GESTIONE BROKER
# ====================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🏦 Gestione broker")

col_add, col_remove = st.sidebar.columns(2)

with col_add:
    if st.button("➕ Aggiungi broker"):
        nuovo_nome = f"Broker {len(st.session_state.brokers_config) + 1}"
        st.session_state.brokers_config[nuovo_nome] = {
            asset: {"min": 2.0, "pct": 0.20, "max": None}
            for asset in ASSET_CLASSES
        }
        st.rerun()

with col_remove:
    if (
        st.button("❌ Rimuovi ultimo")
        and len(st.session_state.brokers_config) > 1
    ):
        ultimo = list(st.session_state.brokers_config.keys())[-1]
        del st.session_state.brokers_config[ultimo]
        st.rerun()

st.sidebar.markdown("""
> 💡 **Nota:**  
> • Commissione **fissa** → minimo = massimo  
> • **Nessun massimo** → lascia il campo vuoto  
""")

# ====================================================
# 🧾 FORM COMMISSIONI (DINAMICO)
# ====================================================

brokers = []

for broker_id, data in st.session_state.brokers_config.items():
    cfg = data[asset_class]

    st.sidebar.subheader(broker_id)

    nome = st.sidebar.text_input(
        "Nome broker/banca",
        value=broker_id,
        key=f"name_{broker_id}"
    )

    minimo = st.sidebar.number_input(
        "Minimo (€)",
        value=cfg["min"],
        step=0.1,
        key=f"min_{broker_id}_{asset_class}"
    )

    percentuale = st.sidebar.number_input(
        "Percentuale (%)",
        value=cfg["pct"],
        step=0.01,
        key=f"pct_{broker_id}_{asset_class}"
    )

    massimo_input = st.sidebar.text_input(
        "Massimo (€) (vuoto = nessun limite)",
        value="" if cfg["max"] is None else str(cfg["max"]),
        key=f"max_{broker_id}_{asset_class}"
    )

    massimo = float(massimo_input) if massimo_input.strip() else None

    st.session_state.brokers_config[broker_id][asset_class] = {
        "min": minimo,
        "pct": percentuale,
        "max": massimo
    }

    brokers.append({
        "nome": nome,
        "minimo": minimo,
        "percentuale": percentuale / 100,
        "massimo": massimo
    })

# ====================================================
# 🧮 FUNZIONE COMMISSIONE
# ====================================================

def calcola_commissione(importo, minimo, percentuale, massimo):
    costo = max(minimo, percentuale * importo)
    if massimo is not None:
        costo = min(costo, massimo)
    return costo

# ====================================================
# 💰 CALCOLO COMMISSIONI (CURVE)
# ====================================================

importi = np.arange(100, 30001, 100)
df = pd.DataFrame({"Importo": importi})

for b in brokers:
    df[b["nome"]] = [
        calcola_commissione(x, b["minimo"], b["percentuale"], b["massimo"])
        for x in df["Importo"]
    ]

# ====================================================
# 🥇 PODIO SINGOLA OPERAZIONE
# ====================================================

st.subheader("🥇 Classifica dei costi per il tuo importo medio")

podio = []
for b in brokers:
    costo = calcola_commissione(
        importo_medio,
        b["minimo"],
        b["percentuale"],
        b["massimo"]
    )
    podio.append((b["nome"], round(costo, 2)))

podio_df = pd.DataFrame(
    podio, columns=["Broker/Banca", "Costo (€)"]
).sort_values("Costo (€)")

best = podio_df.iloc[0]
worst = podio_df.iloc[-1]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🏆 Più economico")
    st.markdown(
        f"<div style='background:#00b050;color:white;padding:8px;border-radius:6px;width:fit-content;'>"
        f"{best['Costo (€)']} €</div>",
        unsafe_allow_html=True
    )
    st.markdown(f"**{best['Broker/Banca']}**")

with col2:
    st.metric("💰 Importo medio", f"{importo_medio} €")

with col3:
    st.markdown("#### 🚫 Meno economico")
    st.markdown(
        f"<div style='background:#ff4b4b;color:white;padding:8px;border-radius:6px;width:fit-content;'>"
        f"{worst['Costo (€)']} €</div>",
        unsafe_allow_html=True
    )
    st.markdown(f"**{worst['Broker/Banca']}**")

st.markdown("---")
st.table(podio_df.set_index("Broker/Banca"))

# ====================================================
# 📊 FASCE DI CONVENIENZA
# ====================================================

st.subheader("📊 Fasce di convenienza")

intersezioni = []

for i in range(len(brokers)):
    for j in range(i + 1, len(brokers)):
        b1, b2 = brokers[i]["nome"], brokers[j]["nome"]
        diff = df[b1] - df[b2]
        segno = np.sign(diff)
        cambi = np.where(np.diff(segno) != 0)[0]

        for c in cambi:
            x1, x2 = df["Importo"].iloc[c], df["Importo"].iloc[c + 1]
            y1, y2 = diff.iloc[c], diff.iloc[c + 1]
            if y2 != y1:
                intersezioni.append(
                    x1 - y1 * (x2 - x1) / (y2 - y1)
                )

intersezioni = sorted(set(intersezioni))
fasce = [0] + intersezioni + [df["Importo"].max()]

records = []
start = fasce[0]
prev_best = None

for i in range(len(fasce) - 1):
    subset = df[
        (df["Importo"] >= fasce[i]) &
        (df["Importo"] < fasce[i + 1])
    ]
    best = subset[[b["nome"] for b in brokers]].mean().idxmin()

    if best != prev_best and prev_best is not None:
        records.append(
            [asset_class, f"{int(start)}–{int(fasce[i])}", prev_best]
        )
        start = fasce[i]

    prev_best = best

records.append(
    [asset_class, f">{int(start)}", prev_best]
)

fasce_df = pd.DataFrame(
    records,
    columns=["Asset Class", "Fascia importo (€)", "Broker/Banca più economico"]
)

st.dataframe(fasce_df)

# ====================================================
# 📈 GRAFICO
# ====================================================

st.subheader("📈 Confronto grafico delle commissioni")

fig = go.Figure()
for b in brokers:
    fig.add_trace(go.Scatter(
        x=df["Importo"],
        y=df[b["nome"]],
        mode="lines",
        name=b["nome"]
    ))

fig.update_layout(
    title=f"Commissioni – {asset_class}",
    xaxis_title="Importo (€)",
    yaxis_title="Commissione (€)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ====================================================
# 🧠 CONSIGLIATORE ANNUALE
# ====================================================

st.markdown("---")
st.subheader("🧠 Consigliatore annuale – costo stimato annuo")

st.sidebar.markdown("---")
st.sidebar.header("🧠 Consigliatore annuale")
st.markdown("""
Per ottenere una stima annua corretta e confrontabile, è necessario compilare i costi per tutte le asset class per tutti i broker.
Asset class o broker lasciati incompleti possono portare a risultati parziali o distorti.
""")
annual_plan = {}

for asset in ASSET_CLASSES:
    st.sidebar.subheader(asset)

    ops = st.sidebar.number_input(
        "Numero operazioni annue",
        min_value=0,
        value=10,
        step=1,
        key=f"ops_{asset}"
    )

    imp = st.sidebar.number_input(
        "Importo medio (€)",
        min_value=0,
        value=2000,
        step=100,
        key=f"ann_importo_{asset}"
    )

    annual_plan[asset] = {"ops": ops, "importo": imp}

annual_results = []

for idx, (broker_id, broker_data) in enumerate(st.session_state.brokers_config.items()):
    totale = 0

    for asset, plan in annual_plan.items():
        if plan["ops"] == 0:
            continue

        cfg = broker_data[asset]

        costo_singola = calcola_commissione(
            plan["importo"],
            cfg["min"],
            cfg["pct"] / 100,
            cfg["max"]
        )

        totale += costo_singola * plan["ops"]

    # Usa il nome personalizzato preso dalla lista 'brokers'
    annual_results.append({
        "Broker/Banca": brokers[idx]["nome"],
        "Costo annuo stimato (€)": round(totale, 2)
    })


annual_df = pd.DataFrame(annual_results).sort_values(
    "Costo annuo stimato (€)"
)

st.dataframe(annual_df, use_container_width=True)

st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
        © 2025, Luca Merlini
    </div>
    """,
    unsafe_allow_html=True
)
