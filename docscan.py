import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


# ============================================================
# FUNÇÕES DE PROCESSAMENTO DE IMAGEM
# ============================================================

def converter_para_cinza(imagem_rgb):
    """
    Converte a imagem colorida RGB para escala de cinza.
    """
    return cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)


def equalizacao_histograma(imagem_cinza):
    """
    Melhora o contraste da imagem usando equalização de histograma.
    """
    return cv2.equalizeHist(imagem_cinza)


def ajustar_brilho_contraste(imagem, brilho=25, contraste=1.4):
    """
    Ajusta brilho e contraste da imagem.

    brilho:
        Valor positivo clareia.
        Valor negativo escurece.

    contraste:
        Valores acima de 1 aumentam o contraste.
        Valores abaixo de 1 reduzem o contraste.
    """
    return cv2.convertScaleAbs(imagem, alpha=contraste, beta=brilho)


def filtro_gaussiano(imagem, tamanho_kernel=5):
    """
    Aplica filtro Gaussiano para suavização e redução de ruído.
    """
    if tamanho_kernel % 2 == 0:
        tamanho_kernel += 1

    if tamanho_kernel < 3:
        tamanho_kernel = 3

    return cv2.GaussianBlur(imagem, (tamanho_kernel, tamanho_kernel), 0)


def detectar_bordas_canny(imagem, limiar1=80, limiar2=160):
    """
    Aplica detecção de bordas usando Canny.
    """
    return cv2.Canny(imagem, limiar1, limiar2)


def limiarizacao_threshold(imagem_cinza, valor=127):
    """
    Aplica threshold global.
    Pixels acima do valor viram branco.
    Pixels abaixo do valor viram preto.
    """
    _, resultado = cv2.threshold(
        imagem_cinza,
        valor,
        255,
        cv2.THRESH_BINARY
    )
    return resultado


def limiarizacao_otsu(imagem_cinza):
    """
    Aplica limiarização automática pelo método de Otsu.
    O próprio algoritmo escolhe o melhor valor de corte.
    """
    _, resultado = cv2.threshold(
        imagem_cinza,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return resultado


def limiarizacao_adaptativa(imagem_cinza, bloco=31, constante=10):
    """
    Aplica limiarização adaptativa.
    Boa para documentos com sombra ou iluminação irregular.
    """
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
    """
    Aplica operação morfológica leve para remover pequenos ruídos.
    """
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
    """
    Pipeline final para gerar uma imagem estilo scanner:
    fundo mais claro e texto mais legível.
    """

    # 1. Equalização de histograma para melhorar contraste
    equalizada = equalizacao_histograma(imagem_cinza)

    # 2. Ajuste de brilho e contraste
    ajustada = ajustar_brilho_contraste(
        equalizada,
        brilho=brilho,
        contraste=contraste
    )

    # 3. Suavização para reduzir ruído
    suavizada = filtro_gaussiano(
        ajustada,
        tamanho_kernel=tamanho_kernel
    )

    # 4. Limiarização adaptativa para destacar texto e clarear fundo
    final = limiarizacao_adaptativa(
        suavizada,
        bloco=bloco,
        constante=constante
    )

    # 5. Limpeza morfológica leve
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
    """
    Processa o documento e retorna todas as etapas do pipeline.
    """

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
        "cinza": cinza,
        "equalizada": equalizada,
        "brilho_contraste": brilho_contraste,
        "suavizada": suavizada,
        "bordas": bordas,
        "threshold_global": threshold_global,
        "otsu": otsu,
        "adaptativa": adaptativa,
        "final": final
    }


def converter_para_download(imagem):
    """
    Converte imagem NumPy para PNG baixável.
    """
    imagem_pil = Image.fromarray(imagem)
    buffer = BytesIO()
    imagem_pil.save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.set_page_config(
    page_title="DOCSCAN PI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DOCSCAN PI")
st.write(
    "Ferramenta de digitalização e melhoria de documentos usando "
    "Processamento de Imagens com Python, OpenCV e Streamlit."
)

st.markdown("---")

# Sidebar com controles
st.sidebar.header("Ajustes do processamento")

brilho = st.sidebar.slider(
    "Brilho",
    min_value=-100,
    max_value=100,
    value=25
)

contraste = st.sidebar.slider(
    "Contraste",
    min_value=0.5,
    max_value=3.0,
    value=1.4,
    step=0.1
)

tamanho_kernel = st.sidebar.slider(
    "Filtro Gaussiano",
    min_value=3,
    max_value=15,
    value=5,
    step=2
)

threshold_valor = st.sidebar.slider(
    "Threshold global",
    min_value=0,
    max_value=255,
    value=127
)

canny_limiar1 = st.sidebar.slider(
    "Canny - Limiar inferior",
    min_value=0,
    max_value=255,
    value=80
)

canny_limiar2 = st.sidebar.slider(
    "Canny - Limiar superior",
    min_value=0,
    max_value=255,
    value=160
)

bloco = st.sidebar.slider(
    "Bloco da limiarização adaptativa",
    min_value=3,
    max_value=99,
    value=31,
    step=2
)

constante = st.sidebar.slider(
    "Constante da limiarização adaptativa",
    min_value=0,
    max_value=30,
    value=10
)


arquivo = st.file_uploader(
    "Carregue uma imagem de documento",
    type=["jpg", "jpeg", "png"]
)


if arquivo is not None:
    imagem_pil = Image.open(arquivo).convert("RGB")
    imagem_rgb = np.array(imagem_pil)

    etapas = processar_documento(
        imagem_rgb,
        brilho=brilho,
        contraste=contraste,
        threshold_valor=threshold_valor,
        tamanho_kernel=tamanho_kernel,
        canny_limiar1=canny_limiar1,
        canny_limiar2=canny_limiar2,
        bloco=bloco,
        constante=constante
    )

    st.subheader("Comparação antes e depois")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            imagem_rgb,
            caption="Imagem original",
            use_container_width=True
        )

    with col2:
        st.image(
            etapas["final"],
            caption="Imagem final processada - estilo scanner",
            use_container_width=True,
            channels="GRAY"
        )

    st.download_button(
        label="Baixar imagem final em PNG",
        data=converter_para_download(etapas["final"]),
        file_name="documento_processado.png",
        mime="image/png"
    )

    st.markdown("---")

    st.subheader("Etapas do processamento")

    col3, col4 = st.columns(2)

    with col3:
        st.image(
            etapas["cinza"],
            caption="1. Escala de cinza",
            use_container_width=True,
            channels="GRAY"
        )

        st.image(
            etapas["equalizada"],
            caption="2. Equalização de histograma",
            use_container_width=True,
            channels="GRAY"
        )

        st.image(
            etapas["threshold_global"],
            caption="5. Threshold global",
            use_container_width=True,
            channels="GRAY"
        )

        st.image(
            etapas["adaptativa"],
            caption="7. Limiarização adaptativa",
            use_container_width=True,
            channels="GRAY"
        )

    with col4:
        st.image(
            etapas["brilho_contraste"],
            caption="3. Ajuste de brilho e contraste",
            use_container_width=True,
            channels="GRAY"
        )

        st.image(
            etapas["suavizada"],
            caption="4. Filtro Gaussiano",
            use_container_width=True,
            channels="GRAY"
        )

        st.image(
            etapas["otsu"],
            caption="6. Limiarização de Otsu",
            use_container_width=True,
            channels="GRAY"
        )

        st.image(
            etapas["bordas"],
            caption="8. Detecção de bordas - Canny",
            use_container_width=True,
            channels="GRAY"
        )

    st.success("Processamento concluído com sucesso.")

else:
    st.info("Envie uma imagem para iniciar o processamento.")
