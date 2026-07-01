from io import BytesIO
import zipfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# UTILIDADES
# ============================================================

def garantir_kernel_impar(valor, minimo=3):
    valor = int(valor)
    if valor < minimo:
        valor = minimo
    if valor % 2 == 0:
        valor += 1
    return valor


def rgb_para_cinza(imagem_rgb):
    return cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)


def imagem_para_png(imagem):
    buffer = BytesIO()

    if len(imagem.shape) == 2:
        imagem_pil = Image.fromarray(imagem)
    else:
        imagem_pil = Image.fromarray(imagem)

    imagem_pil.save(buffer, format="PNG")
    return buffer.getvalue()


def imagem_para_pdf(imagem):
    buffer = BytesIO()

    if len(imagem.shape) == 2:
        imagem_pil = Image.fromarray(imagem).convert("RGB")
    else:
        imagem_pil = Image.fromarray(imagem).convert("RGB")

    imagem_pil.save(buffer, format="PDF", resolution=100.0)
    return buffer.getvalue()


def gerar_zip(etapas):
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for nome, imagem in etapas.items():
            zip_file.writestr(f"{nome}.png", imagem_para_png(imagem))

        zip_file.writestr(
            "resultado_final.pdf",
            imagem_para_pdf(etapas["07_resultado_final"])
        )

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# DETECCAO E CORRECAO DO DOCUMENTO
# ============================================================

def ordenar_pontos(pontos):
    """
    Ordena os quatro pontos do documento em:
    superior esquerdo, superior direito, inferior direito, inferior esquerdo.
    """
    pontos = pontos.reshape(4, 2).astype("float32")

    retangulo = np.zeros((4, 2), dtype="float32")

    soma = pontos.sum(axis=1)
    diferenca = np.diff(pontos, axis=1)

    retangulo[0] = pontos[np.argmin(soma)]
    retangulo[2] = pontos[np.argmax(soma)]
    retangulo[1] = pontos[np.argmin(diferenca)]
    retangulo[3] = pontos[np.argmax(diferenca)]

    return retangulo


def transformar_perspectiva(imagem, pontos):
    """
    Aplica transformacao de perspectiva para deixar o documento frontal,
    centralizado e recortado.
    """
    retangulo = ordenar_pontos(pontos)
    sup_esq, sup_dir, inf_dir, inf_esq = retangulo

    largura_a = np.linalg.norm(inf_dir - inf_esq)
    largura_b = np.linalg.norm(sup_dir - sup_esq)
    largura = int(max(largura_a, largura_b))

    altura_a = np.linalg.norm(sup_dir - inf_dir)
    altura_b = np.linalg.norm(sup_esq - inf_esq)
    altura = int(max(altura_a, altura_b))

    destino = np.array([
        [0, 0],
        [largura - 1, 0],
        [largura - 1, altura - 1],
        [0, altura - 1]
    ], dtype="float32")

    matriz = cv2.getPerspectiveTransform(retangulo, destino)
    documento = cv2.warpPerspective(imagem, matriz, (largura, altura))

    return documento


def detectar_documento(imagem_rgb, canny_min=50, canny_max=150):
    """
    Detecta o maior contorno quadrilateral compatível com documento.
    Se falhar, retorna a imagem original como fallback.
    """
    altura_original = imagem_rgb.shape[0]
    escala = altura_original / 700.0

    largura_redimensionada = int(imagem_rgb.shape[1] / escala)
    imagem_pequena = cv2.resize(imagem_rgb, (largura_redimensionada, 700))

    cinza = rgb_para_cinza(imagem_pequena)
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)

    bordas = cv2.Canny(suavizada, canny_min, canny_max)

    kernel = np.ones((3, 3), np.uint8)
    bordas_dilatadas = cv2.dilate(bordas, kernel, iterations=1)

    contornos, _ = cv2.findContours(
        bordas_dilatadas,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    area_imagem = imagem_pequena.shape[0] * imagem_pequena.shape[1]
    area_minima = area_imagem * 0.08

    for contorno in contornos:
        area = cv2.contourArea(contorno)

        if area < area_minima:
            continue

        perimetro = cv2.arcLength(contorno, True)
        aproximado = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)

        if len(aproximado) == 4:
            pontos_originais = aproximado.reshape(4, 2).astype("float32") * escala
            documento = transformar_perspectiva(imagem_rgb, pontos_originais)

            return {
                "documento": documento,
                "detectado": True,
                "bordas": bordas,
                "bordas_dilatadas": bordas_dilatadas,
                "pontos": pontos_originais
            }

    return {
        "documento": imagem_rgb,
        "detectado": False,
        "bordas": bordas,
        "bordas_dilatadas": bordas_dilatadas,
        "pontos": None
    }


# ============================================================
# MELHORIA DE LEGIBILIDADE
# ============================================================

def corrigir_iluminacao(imagem_cinza):
    """
    Reduz sombras e iluminacao irregular usando fundo estimado por blur grande.
    """
    fundo = cv2.GaussianBlur(imagem_cinza, (0, 0), sigmaX=35, sigmaY=35)

    corrigida = cv2.divide(imagem_cinza, fundo, scale=255)
    corrigida = cv2.normalize(corrigida, None, 0, 255, cv2.NORM_MINMAX)

    return corrigida.astype("uint8")


def melhorar_legibilidade(imagem_rgb, brilho=8, contraste=1.15):
    """
    Gera versao em cinza aprimorada preservando detalhes finos.
    Usa correcao de iluminacao, CLAHE, brilho/contraste e suavizacao leve.
    """
    cinza = rgb_para_cinza(imagem_rgb)

    sem_sombra = corrigir_iluminacao(cinza)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    melhorada = clahe.apply(sem_sombra)

    melhorada = cv2.convertScaleAbs(
        melhorada,
        alpha=float(contraste),
        beta=int(brilho)
    )

    melhorada = cv2.bilateralFilter(melhorada, 7, 35, 35)

    return melhorada


def gerar_scanner_pb(imagem_cinza, bloco=31, constante=11):
    """
    Gera uma versao preto e branco estilo scanner.
    Evita agressividade excessiva aplicando suavizacao e morfologia leve.
    """
    bloco = garantir_kernel_impar(bloco, minimo=3)

    suavizada = cv2.GaussianBlur(imagem_cinza, (3, 3), 0)

    otsu = cv2.threshold(
        suavizada,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    adaptativa = cv2.adaptiveThreshold(
        suavizada,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bloco,
        constante
    )

    combinada = cv2.bitwise_and(otsu, adaptativa)

    kernel = np.ones((2, 2), np.uint8)
    limpa = cv2.morphologyEx(combinada, cv2.MORPH_OPEN, kernel)

    return limpa


def calcular_score_colorido(imagem_rgb):
    """
    Estima se o documento possui muitos detalhes coloridos.
    CNH/RG/cartoes tendem a ter saturacao maior que folhas simples.
    """
    hsv = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2HSV)
    saturacao_media = hsv[:, :, 1].mean()

    return saturacao_media


def escolher_resultado_automatico(imagem_rgb, cinza_aprimorado, scanner_pb):
    """
    Modo Automatico inteligente:
    - documentos coloridos/detalhados: prefere cinza aprimorado;
    - folhas simples com texto escuro: prefere scanner P&B se a binarizacao ficou saudavel.
    """
    score_cor = calcular_score_colorido(imagem_rgb)
    proporcao_preto = np.mean(scanner_pb < 128)

    documento_colorido = score_cor > 35
    binarizacao_saudavel = 0.03 <= proporcao_preto <= 0.45

    if documento_colorido:
        return cinza_aprimorado, "Cinza aprimorado"

    if binarizacao_saudavel:
        return scanner_pb, "Scanner P&B"

    return cinza_aprimorado, "Cinza aprimorado"


def gerar_resultado_final(
    documento_rgb,
    modo="Automático inteligente",
    brilho=8,
    contraste=1.15,
    bloco=31,
    constante=11
):
    """
    Gera as tres versoes principais:
    1. documento recortado/colorido;
    2. cinza aprimorado;
    3. scanner preto e branco suave.
    """
    cinza_aprimorado = melhorar_legibilidade(
        documento_rgb,
        brilho=brilho,
        contraste=contraste
    )

    scanner_pb = gerar_scanner_pb(
        cinza_aprimorado,
        bloco=bloco,
        constante=constante
    )

    if modo == "Colorido recortado":
        final = documento_rgb
        nome_final = "Colorido recortado"

    elif modo == "Cinza aprimorado":
        final = cinza_aprimorado
        nome_final = "Cinza aprimorado"

    elif modo == "Scanner P&B":
        final = scanner_pb
        nome_final = "Scanner P&B"

    else:
        final, nome_final = escolher_resultado_automatico(
            documento_rgb,
            cinza_aprimorado,
            scanner_pb
        )

    return {
        "colorido": documento_rgb,
        "cinza_aprimorado": cinza_aprimorado,
        "scanner_pb": scanner_pb,
        "final": final,
        "modo_final": nome_final
    }


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="DOCSCAN PI",
    page_icon="📄",
    layout="wide"
)

st.title("DOCSCAN PI")
st.write(
    "Scanner automático de documentos com OpenCV e Streamlit. "
    "A aplicação detecta a folha, corrige a perspectiva e só depois aplica "
    "os filtros de melhoria."
)

st.sidebar.header("Modo de saída")

modo_saida = st.sidebar.selectbox(
    "Resultado final",
    [
        "Automático inteligente",
        "Colorido recortado",
        "Cinza aprimorado",
        "Scanner P&B"
    ]
)

st.sidebar.header("Ajustes avançados")

with st.sidebar.expander("Detecção do documento", expanded=False):
    canny_min = st.slider("Canny mínimo", 0, 255, 50)
    canny_max = st.slider("Canny máximo", 0, 255, 150)

with st.sidebar.expander("Legibilidade", expanded=False):
    brilho = st.slider("Brilho", -50, 80, 8)
    contraste = st.slider("Contraste", 0.6, 2.0, 1.15, 0.05)

with st.sidebar.expander("Scanner P&B", expanded=False):
    bloco = st.slider("Bloco adaptativo", 3, 99, 31, step=2)
    constante = st.slider("Constante adaptativa", 0, 30, 11)

arquivo = st.file_uploader(
    "Carregue uma foto de documento",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if arquivo is None:
    st.info("Envie uma imagem para iniciar o processamento.")
    st.stop()

imagem_pil = Image.open(arquivo).convert("RGB")
imagem_rgb = np.array(imagem_pil)

detecao = detectar_documento(
    imagem_rgb,
    canny_min=canny_min,
    canny_max=canny_max
)

documento_rgb = detecao["documento"]

if detecao["detectado"]:
    st.success("Documento detectado automaticamente. Perspectiva corrigida.")
else:
    st.warning(
        "Não foi possível detectar automaticamente as bordas do documento. "
        "Tente uma foto com melhor contraste entre documento e fundo."
    )

resultado = gerar_resultado_final(
    documento_rgb,
    modo=modo_saida,
    brilho=brilho,
    contraste=contraste,
    bloco=bloco,
    constante=constante
)

st.subheader("Comparação principal")

col1, col2 = st.columns(2)

with col1:
    st.image(imagem_rgb, caption="Imagem original", use_container_width=True)

with col2:
    st.image(
        resultado["final"],
        caption=f"Resultado final - {resultado['modo_final']}",
        use_container_width=True
    )

st.subheader("Pipeline automático")

etapas = {
    "01_original": imagem_rgb,
    "02_bordas_canny": detecao["bordas"],
    "03_bordas_dilatadas": detecao["bordas_dilatadas"],
    "04_documento_recortado_colorido": resultado["colorido"],
    "05_cinza_aprimorado": resultado["cinza_aprimorado"],
    "06_scanner_preto_e_branco": resultado["scanner_pb"],
    "07_resultado_final": resultado["final"],
}

titulos = {
    "01_original": "Imagem original",
    "02_bordas_canny": "Canny para detecção",
    "03_bordas_dilatadas": "Bordas dilatadas",
    "04_documento_recortado_colorido": "Documento recortado / perspectiva",
    "05_cinza_aprimorado": "Cinza aprimorado",
    "06_scanner_preto_e_branco": "Scanner P&B suave",
    "07_resultado_final": "Resultado final"
}

colunas = st.columns(3)

for indice, (nome, imagem) in enumerate(etapas.items()):
    with colunas[indice % 3]:
        st.markdown(f"**{titulos[nome]}**")
        st.image(imagem, use_container_width=True)

        st.download_button(
            label="Baixar PNG",
            data=imagem_para_png(imagem),
            file_name=f"{nome}.png",
            mime="image/png",
            key=f"download_{nome}"
        )

st.subheader("Downloads")

col_png, col_pdf, col_zip = st.columns(3)

with col_png:
    st.download_button(
        "Baixar resultado final PNG",
        data=imagem_para_png(resultado["final"]),
        file_name="docscan_resultado_final.png",
        mime="image/png"
    )

with col_pdf:
    st.download_button(
        "Baixar resultado final PDF",
        data=imagem_para_pdf(resultado["final"]),
        file_name="docscan_resultado_final.pdf",
        mime="application/pdf"
    )

with col_zip:
    st.download_button(
        "Baixar ZIP com todas as imagens",
        data=gerar_zip(etapas),
        file_name="docscan_pi_resultados.zip",
        mime="application/zip"
    )

st.markdown("---")

st.markdown(
    """
### Fluxo aplicado

Imagem original → pré-processamento → escala de cinza → suavização → Canny → contornos →
maior quadrilátero → perspectiva → documento recortado → correção de iluminação →
melhoria de contraste → scanner final.
"""
)
