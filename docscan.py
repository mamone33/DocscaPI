import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import zipfile


# ============================================================
# CONFIGURAÇÕES PADRÃO DOS SLIDERS
# ============================================================

VALORES_PADRAO = {
    "brilho": 25,
    "contraste": 1.4,
    "tamanho_kernel": 5,
    "threshold_valor": 127,
    "canny_limiar1": 80,
    "canny_limiar2": 160,
    "bloco": 31,
    "constante": 10,
}


def inicializar_sliders():
    for chave, valor in VALORES_PADRAO.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


def resetar_sliders():
    for chave, valor in VALORES_PADRAO.items():
        st.session_state[chave] = valor


# ============================================================
# FUNÇÕES DE PROCESSAMENTO DE IMAGEM
# ============================================================

def converter_para_cinza(imagem_rgb):
    return cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)


def equalizacao_histograma(imagem_cinza):
    return cv2.equalizeHist(imagem_cinza)


def ajustar_brilho_contraste(imagem, brilho=25, contraste=1.4):
    return cv2.convertScaleAbs(imagem, alpha=contraste, beta=brilho)


def filtro_gaussiano(imagem, tamanho_kernel=5):
    if tamanho_kernel % 2 == 0:
        tamanho_kernel += 1

    if tamanho_kernel < 3:
        tamanho_kernel = 3

    return cv2.GaussianBlur(imagem, (tamanho_kernel, tamanho_kernel), 0)


def detectar_bordas_canny(imagem, limiar1=80, limiar2=160):
    return cv2.Canny(imagem, limiar1, limiar2)


def limiarizacao_threshold(imagem_cinza, valor=127):
    _, resultado = cv2.threshold(
        imagem_cinza,
        valor,
        255,
        cv2.THRESH_BINARY
    )
    return resultado


def limiarizacao_otsu(imagem_cinza):
    _, resultado = cv2.threshold(
        imagem_cinza,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return resultado


def limiarizacao_adaptativa(imagem_cinza, bloco=31, constante=10):
    if bloco % 2 == 0:
        bloco += 1

    if bloco < 3:
        bloco = 3

    return cv2.adaptiveThreshold(
        imagem_cinza,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bloco,
        constante
    )


def limpar_imagem_morfologia(imagem):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(imagem, cv2.MORPH_OPEN, kernel)


def gerar_imagem_final_scanner(
    imagem_cinza,
    brilho=25,
    contraste=1.4,
    tamanho_kernel=5,
    bloco=31,
    constante=10
):
    equalizada = equalizacao_histograma(imagem_cinza)

    ajustada = ajustar_brilho_contraste(
        equalizada,
        brilho=brilho,
        contraste=contraste
    )

    suavizada = filtro_gaussiano(
        ajustada,
        tamanho_kernel=tamanho_kernel
    )

    final = limiarizacao_adaptativa(
        suavizada,
        bloco=bloco,
        constante=constante
    )

    final = limpar_imagem_morfologia(final)

    return final


def processar_documento(
    imagem_rgb,
    brilho=25,
    contraste=1.4,
    threshold_valor=127,
    tamanho_kernel=5,
    canny_limiar1=80,
    canny_limiar2=160,
    bloco=31,
    constante=10
):
    cinza = converter_para_cinza(imagem_rgb)

    equalizada = equalizacao_histograma(cinza)

    brilho_contraste = ajustar_brilho_contraste(
        equalizada,
        brilho=brilho,
        contraste=contraste
    )

    suavizada = filtro_gaussiano(
        brilho_contraste,
        tamanho_kernel=tamanho_kernel
    )

    bordas = detectar_bordas_canny(
        suavizada,
        limiar1=canny_limiar1,
        limiar2=canny_limiar2
    )

    threshold_global = limiarizacao_threshold(
        suavizada,
        valor=threshold_valor
    )

    otsu = limiarizacao_otsu(suavizada)

    adaptativa = limiarizacao_adaptativa(
        suavizada,
        bloco=bloco,
        constante=constante
    )

    final = gerar_imagem_final_scanner(
        cinza,
        brilho=brilho,
        contraste=contraste,
        tamanho_kernel=tamanho_kernel,
        bloco=bloco,
        constante=constante
    )

    return {
        "01_original": imagem_rgb,
        "02_escala_cinza": cinza,
        "03_equalizacao_histograma": equalizada,
        "04_brilho_contraste": brilho_contraste,
        "05_filtro_gaussiano": suavizada,
        "06_bordas_canny": bordas,
        "07_threshold_global": threshold_global,
        "08_limiarizacao_otsu": otsu,
        "09_limiarizacao_adaptativa": adaptativa,
        "10_imagem_final_scanner": final
    }


# ============================================================
# FUNÇÕES DE DOWNLOAD
# ============================================================

def converter_para_download(imagem):
    imagem_pil = Image.fromarray(imagem)
    buffer = BytesIO()
    imagem_pil.save(buffer, format="PNG")
    return buffer.getvalue()


def gerar_zip_com_imagens(etapas):
    buffer_zip = BytesIO()

    with zipfile.ZipFile(buffer_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for nome, imagem in etapas.items():
            imagem_png = converter_para_download(imagem)
            zip_file.writestr(f"{nome}.png", imagem_png)

    buffer_zip.seek(0)
    return buffer_zip.getvalue()


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.set_page_config(
    page_title="DOCSCAN PI",
    page_icon="📄",
    layout="wide"
)

inicializar_sliders()

if "arquivo_carregado_anteriormente" not in st.session_state:
    st.session_state.arquivo_carregado_anteriormente = False

st.title("📄 DOCSCAN PI")
st.write(
    "Ferramenta de digitalização e melhoria de documentos usando "
    "Processamento de Imagens com Python, OpenCV e Streamlit."
)

st.markdown("---")

arquivo = st.file_uploader(
    "Carregue uma imagem de documento",
    type=["jpg", "jpeg", "png"],
    key="upload_imagem"
)


# ============================================================
# RESET AUTOMÁTICO DOS SLIDERS AO REMOVER O ARQUIVO
# ============================================================

if arquivo is None and st.session_state.arquivo_carregado_anteriormente:
    resetar_sliders()
    st.session_state.arquivo_carregado_anteriormente = False
    st.rerun()

if arquivo is not None:
    st.session_state.arquivo_carregado_anteriormente = True


# ============================================================
# SIDEBAR COM SLIDERS
# ============================================================

st.sidebar.header("Ajustes do processamento")

st.sidebar.slider(
    "Brilho",
    min_value=-100,
    max_value=100,
    key="brilho"
)

st.sidebar.slider(
    "Contraste",
    min_value=0.5,
    max_value=3.0,
    step=0.1,
    key="contraste"
)

st.sidebar.slider(
    "Filtro Gaussiano",
    min_value=3,
    max_value=15,
    step=2,
    key="tamanho_kernel"
)

st.sidebar.slider(
    "Threshold global",
    min_value=0,
    max_value=255,
    key="threshold_valor"
)

st.sidebar.slider(
    "Canny - Limiar inferior",
    min_value=0,
    max_value=255,
    key="canny_limiar1"
)

st.sidebar.slider(
    "Canny - Limiar superior",
    min_value=0,
    max_value=255,
    key="canny_limiar2"
)

st.sidebar.slider(
    "Bloco da limiarização adaptativa",
    min_value=3,
    max_value=99,
    step=2,
    key="bloco"
)

st.sidebar.slider(
    "Constante da limiarização adaptativa",
    min_value=0,
    max_value=30,
    key="constante"
)


# ============================================================
# PROCESSAMENTO
# ============================================================

if arquivo is not None:
    imagem_pil = Image.open(arquivo).convert("RGB")
    imagem_rgb = np.array(imagem_pil)

    etapas = processar_documento(
        imagem_rgb,
        brilho=st.session_state.brilho,
        contraste=st.session_state.contraste,
        threshold_valor=st.session_state.threshold_valor,
        tamanho_kernel=st.session_state.tamanho_kernel,
        canny_limiar1=st.session_state.canny_limiar1,
        canny_limiar2=st.session_state.canny_limiar2,
        bloco=st.session_state.bloco,
        constante=st.session_state.constante
    )

    st.subheader("Comparação antes e depois")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            etapas["01_original"],
            caption="Imagem original",
            use_container_width=True
        )

        st.download_button(
            label="Baixar imagem original",
            data=converter_para_download(etapas["01_original"]),
            file_name="01_original.png",
            mime="image/png"
        )

    with col2:
        st.image(
            etapas["10_imagem_final_scanner"],
            caption="Imagem final processada - estilo scanner",
            use_container_width=True,
            channels="GRAY"
        )

        st.download_button(
            label="Baixar imagem final",
            data=converter_para_download(etapas["10_imagem_final_scanner"]),
            file_name="10_imagem_final_scanner.png",
            mime="image/png"
        )

    st.markdown("---")

    st.subheader("Download geral")

    st.download_button(
        label="Baixar todas as imagens em ZIP",
        data=gerar_zip_com_imagens(etapas),
        file_name="docscan_resultados.zip",
        mime="application/zip"
    )

    st.markdown("---")

    st.subheader("Etapas do processamento")

    nomes_legiveis = {
        "02_escala_cinza": "1. Escala de cinza",
        "03_equalizacao_histograma": "2. Equalização de histograma",
        "04_brilho_contraste": "3. Ajuste de brilho e contraste",
        "05_filtro_gaussiano": "4. Filtro Gaussiano",
        "06_bordas_canny": "5. Detecção de bordas - Canny",
        "07_threshold_global": "6. Threshold global",
        "08_limiarizacao_otsu": "7. Limiarização de Otsu",
        "09_limiarizacao_adaptativa": "8. Limiarização adaptativa",
        "10_imagem_final_scanner": "9. Resultado final estilo scanner"
    }

    for nome_arquivo, titulo in nomes_legiveis.items():
        st.markdown(f"### {titulo}")

        st.image(
            etapas[nome_arquivo],
            use_container_width=True,
            channels="GRAY"
        )

        st.download_button(
            label=f"Baixar {titulo}",
            data=converter_para_download(etapas[nome_arquivo]),
            file_name=f"{nome_arquivo}.png",
            mime="image/png",
            key=f"download_{nome_arquivo}"
        )

    st.success("Processamento concluído com sucesso.")

else:
    st.info("Envie uma imagem para iniciar o processamento.")
