import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def converter_para_cinza(imagem):
    return cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)


def melhorar_contraste(imagem_cinza):
    return cv2.equalizeHist(imagem_cinza)


def filtro_gaussiano(imagem):
    return cv2.GaussianBlur(imagem, (5, 5), 0)


def detectar_bordas_canny(imagem):
    return cv2.Canny(imagem, 80, 160)


def limiarizacao_adaptativa(imagem):
    return cv2.adaptiveThreshold(
        imagem,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )


def processar_documento(imagem_rgb):
    cinza = converter_para_cinza(imagem_rgb)
    contraste = melhorar_contraste(cinza)
    suavizada = filtro_gaussiano(contraste)
    bordas = detectar_bordas_canny(suavizada)
    final = limiarizacao_adaptativa(suavizada)

    return {
        "cinza": cinza,
        "contraste": contraste,
        "suavizada": suavizada,
        "bordas": bordas,
        "final": final
    }


def converter_para_download(imagem):
    imagem_pil = Image.fromarray(imagem)
    buffer = BytesIO()
    imagem_pil.save(buffer, format="PNG")
    return buffer.getvalue()


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

arquivo = st.file_uploader(
    "Carregue uma imagem de documento",
    type=["jpg", "jpeg", "png"]
)

if arquivo is not None:
    imagem_pil = Image.open(arquivo).convert("RGB")
    imagem_rgb = np.array(imagem_pil)

    etapas = processar_documento(imagem_rgb)

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
            caption="Imagem final processada",
            use_container_width=True,
            channels="GRAY"
        )

    st.download_button(
        label="Baixar imagem final em PNG",
        data=converter_para_download(etapas["final"]),
        file_name="documento_processado.png",
        mime="image/png"
    )

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
            etapas["suavizada"],
            caption="3. Filtro Gaussiano",
            use_container_width=True,
            channels="GRAY"
        )

    with col4:
        st.image(
            etapas["contraste"],
            caption="2. Melhoria de contraste",
            use_container_width=True,
            channels="GRAY"
        )

        st.image(
            etapas["bordas"],
            caption="4. Detecção de bordas - Canny",
            use_container_width=True,
            channels="GRAY"
        )

    st.success("Processamento concluído com sucesso.")

else:
    st.info("Envie uma imagem para iniciar o processamento.")
