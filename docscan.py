from io import BytesIO
import zipfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURAÇÕES PADRÃO
# ============================================================

PADROES = {
    "modo_saida": "Automático inteligente",
    "usar_deteccao": True,
    "canny_min": 50,
    "canny_max": 150,
    "brilho": 8,
    "contraste": 1.15,
    "metodo_suavizacao": "Bilateral",
    "kernel_suavizacao": 5,
    "forca_clahe": 2.0,
    "grid_clahe": 8,
    "bloco": 31,
    "constante": 11,
    "limpeza_morfologica": True,
}


def inicializar_estado():
    for chave, valor in PADROES.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


def resetar_filtros():
    for chave, valor in PADROES.items():
        st.session_state[chave] = valor


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
    Image.fromarray(imagem).save(buffer, format="PNG")
    return buffer.getvalue()


def imagem_para_pdf(imagem):
    buffer = BytesIO()
    imagem_pil = Image.fromarray(imagem)
    if imagem_pil.mode != "RGB":
        imagem_pil = imagem_pil.convert("RGB")
    imagem_pil.save(buffer, format="PDF", resolution=120.0)
    return buffer.getvalue()


def gerar_zip(etapas):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for nome, imagem in etapas.items():
            zip_file.writestr(f"{nome}.png", imagem_para_png(imagem))
        zip_file.writestr("resultado_final.pdf", imagem_para_pdf(etapas["10_resultado_final"]))
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# DETECÇÃO E CORREÇÃO DE PERSPECTIVA
# ============================================================

def ordenar_pontos(pontos):
    pontos = pontos.reshape(4, 2).astype("float32")
    retangulo = np.zeros((4, 2), dtype="float32")

    soma = pontos.sum(axis=1)
    diferenca = np.diff(pontos, axis=1)

    retangulo[0] = pontos[np.argmin(soma)]       # superior esquerdo
    retangulo[2] = pontos[np.argmax(soma)]       # inferior direito
    retangulo[1] = pontos[np.argmin(diferenca)]  # superior direito
    retangulo[3] = pontos[np.argmax(diferenca)]  # inferior esquerdo

    return retangulo


def transformar_perspectiva(imagem, pontos):
    retangulo = ordenar_pontos(pontos)
    sup_esq, sup_dir, inf_dir, inf_esq = retangulo

    largura_a = np.linalg.norm(inf_dir - inf_esq)
    largura_b = np.linalg.norm(sup_dir - sup_esq)
    largura = int(max(largura_a, largura_b))

    altura_a = np.linalg.norm(sup_dir - inf_dir)
    altura_b = np.linalg.norm(sup_esq - inf_esq)
    altura = int(max(altura_a, altura_b))

    largura = max(largura, 100)
    altura = max(altura, 100)

    destino = np.array(
        [[0, 0], [largura - 1, 0], [largura - 1, altura - 1], [0, altura - 1]],
        dtype="float32",
    )

    matriz = cv2.getPerspectiveTransform(retangulo, destino)
    return cv2.warpPerspective(imagem, matriz, (largura, altura))


def detectar_documento(imagem_rgb, canny_min=50, canny_max=150):
    """
    Detecta o maior contorno quadrilateral compatível com documento.
    Se falhar, retorna a imagem original como fallback.
    """
    altura_original = imagem_rgb.shape[0]
    escala = altura_original / 700.0 if altura_original > 700 else 1.0

    if escala != 1.0:
        largura_redimensionada = int(imagem_rgb.shape[1] / escala)
        imagem_pequena = cv2.resize(imagem_rgb, (largura_redimensionada, 700))
    else:
        imagem_pequena = imagem_rgb.copy()

    cinza = rgb_para_cinza(imagem_pequena)
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)
    bordas = cv2.Canny(suavizada, canny_min, canny_max)

    kernel = np.ones((3, 3), np.uint8)
    bordas_dilatadas = cv2.dilate(bordas, kernel, iterations=1)

    contornos, _ = cv2.findContours(
        bordas_dilatadas,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    area_imagem = imagem_pequena.shape[0] * imagem_pequena.shape[1]
    area_minima = area_imagem * 0.05

    for contorno in contornos[:15]:
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
                "pontos": pontos_originais,
            }

    return {
        "documento": imagem_rgb,
        "detectado": False,
        "bordas": bordas,
        "bordas_dilatadas": bordas_dilatadas,
        "pontos": None,
    }


# ============================================================
# FILTROS E MELHORIAS
# ============================================================

def equalizacao_histograma(imagem_cinza):
    return cv2.equalizeHist(imagem_cinza)


def aplicar_clahe(imagem_cinza, clip_limit=2.0, grid_size=8):
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(grid_size), int(grid_size)),
    )
    return clahe.apply(imagem_cinza)


def ajustar_brilho_contraste(imagem, brilho=8, contraste=1.15):
    return cv2.convertScaleAbs(imagem, alpha=float(contraste), beta=int(brilho))


def corrigir_iluminacao(imagem_cinza):
    """Reduz sombras e iluminação irregular estimando o fundo com blur grande."""
    fundo = cv2.GaussianBlur(imagem_cinza, (0, 0), sigmaX=35, sigmaY=35)
    corrigida = cv2.divide(imagem_cinza, fundo, scale=255)
    corrigida = cv2.normalize(corrigida, None, 0, 255, cv2.NORM_MINMAX)
    return corrigida.astype("uint8")


def filtro_gaussiano(imagem, tamanho_kernel=5):
    k = garantir_kernel_impar(tamanho_kernel)
    return cv2.GaussianBlur(imagem, (k, k), 0)


def filtro_mediana(imagem, tamanho_kernel=5):
    k = garantir_kernel_impar(tamanho_kernel)
    return cv2.medianBlur(imagem, k)


def filtro_bilateral(imagem):
    return cv2.bilateralFilter(imagem, 9, 75, 75)


def aplicar_suavizacao(imagem, metodo="Bilateral", tamanho_kernel=5):
    if metodo == "Gaussiano":
        return filtro_gaussiano(imagem, tamanho_kernel)
    if metodo == "Mediana":
        return filtro_mediana(imagem, tamanho_kernel)
    if metodo == "Bilateral":
        return filtro_bilateral(imagem)
    return imagem


def detectar_bordas_canny(imagem, limiar1=80, limiar2=160):
    return cv2.Canny(imagem, limiar1, limiar2)


def limiarizacao_threshold(imagem_cinza, valor=127):
    _, resultado = cv2.threshold(imagem_cinza, valor, 255, cv2.THRESH_BINARY)
    return resultado


def limiarizacao_otsu(imagem_cinza):
    _, resultado = cv2.threshold(
        imagem_cinza,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return resultado


def limiarizacao_adaptativa(imagem_cinza, bloco=31, constante=11):
    bloco = garantir_kernel_impar(bloco)
    return cv2.adaptiveThreshold(
        imagem_cinza,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bloco,
        int(constante),
    )


def limpar_binaria(imagem_binaria):
    kernel_pequeno = np.ones((2, 2), np.uint8)
    limpa = cv2.morphologyEx(imagem_binaria, cv2.MORPH_OPEN, kernel_pequeno)
    limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, kernel_pequeno)
    return limpa


def scanner_pb_suave(imagem_suavizada, bloco=31, constante=11, limpar=True):
    otsu = limiarizacao_otsu(imagem_suavizada)
    adaptativa = limiarizacao_adaptativa(imagem_suavizada, bloco, constante)

    # Combinação menos agressiva que a anterior: preserva melhor textos finos.
    proporcao_preto_adapt = np.mean(adaptativa < 128)
    if 0.03 <= proporcao_preto_adapt <= 0.45:
        resultado = adaptativa
    else:
        resultado = otsu

    if limpar:
        resultado = limpar_binaria(resultado)

    return resultado, otsu, adaptativa


def calcular_score_colorido(imagem_rgb):
    hsv = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 1].mean()


def escolher_resultado_automatico(documento_rgb, cinza_aprimorado, scanner_pb):
    score_cor = calcular_score_colorido(documento_rgb)
    proporcao_preto = np.mean(scanner_pb < 128)

    documento_colorido = score_cor > 35
    binarizacao_saudavel = 0.03 <= proporcao_preto <= 0.45

    if documento_colorido:
        return cinza_aprimorado, "Cinza aprimorado"
    if binarizacao_saudavel:
        return scanner_pb, "Scanner P&B"
    return cinza_aprimorado, "Cinza aprimorado"


def processar_documento(
    imagem_rgb,
    usar_deteccao=True,
    modo_saida="Automático inteligente",
    canny_min=50,
    canny_max=150,
    brilho=8,
    contraste=1.15,
    metodo_suavizacao="Bilateral",
    kernel_suavizacao=5,
    forca_clahe=2.0,
    grid_clahe=8,
    bloco=31,
    constante=11,
    limpeza_morfologica=True,
):
    if usar_deteccao:
        deteccao = detectar_documento(imagem_rgb, canny_min, canny_max)
        documento_rgb = deteccao["documento"]
    else:
        deteccao = {
            "documento": imagem_rgb,
            "detectado": False,
            "bordas": detectar_bordas_canny(rgb_para_cinza(imagem_rgb), canny_min, canny_max),
            "bordas_dilatadas": detectar_bordas_canny(rgb_para_cinza(imagem_rgb), canny_min, canny_max),
            "pontos": None,
        }
        documento_rgb = imagem_rgb

    cinza = rgb_para_cinza(documento_rgb)
    sem_sombra = corrigir_iluminacao(cinza)
    equalizada = equalizacao_histograma(sem_sombra)
    clahe = aplicar_clahe(sem_sombra, forca_clahe, grid_clahe)
    brilho_contraste = ajustar_brilho_contraste(clahe, brilho, contraste)
    suavizada = aplicar_suavizacao(brilho_contraste, metodo_suavizacao, kernel_suavizacao)
    bordas_processamento = detectar_bordas_canny(suavizada, canny_min, canny_max)
    threshold_global = limiarizacao_threshold(suavizada, 127)
    scanner_pb, otsu, adaptativa = scanner_pb_suave(
        suavizada,
        bloco=bloco,
        constante=constante,
        limpar=limpeza_morfologica,
    )

    cinza_aprimorado = suavizada

    if modo_saida == "Colorido recortado":
        final = documento_rgb
        modo_final = "Colorido recortado"
    elif modo_saida == "Cinza aprimorado":
        final = cinza_aprimorado
        modo_final = "Cinza aprimorado"
    elif modo_saida == "Scanner P&B":
        final = scanner_pb
        modo_final = "Scanner P&B"
    else:
        final, modo_final = escolher_resultado_automatico(documento_rgb, cinza_aprimorado, scanner_pb)

    etapas = {
        "01_original": imagem_rgb,
        "02_bordas_para_detectar": deteccao["bordas"],
        "03_bordas_dilatadas": deteccao["bordas_dilatadas"],
        "04_documento_recortado_colorido": documento_rgb,
        "05_escala_cinza": cinza,
        "06_correcao_iluminacao": sem_sombra,
        "07_equalizacao_histograma": equalizada,
        "08_contraste_clahe_brilho": brilho_contraste,
        "09_suavizacao": suavizada,
        "10_resultado_final": final,
        "11_scanner_preto_e_branco": scanner_pb,
        "12_limiarizacao_otsu": otsu,
        "13_limiarizacao_adaptativa": adaptativa,
        "14_bordas_canny_processamento": bordas_processamento,
        "15_threshold_global": threshold_global,
    }

    return etapas, deteccao, modo_final


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.set_page_config(page_title="DOCSCAN PI", page_icon="📄", layout="wide")
inicializar_estado()

st.title("📄 DOCSCAN PI")
st.write(
    "Scanner automático de documentos com OpenCV e Streamlit. "
    "A aplicação detecta o documento, corrige a perspectiva e depois permite "
    "ajustar filtros de legibilidade."
)
st.markdown("---")

# Upload primeiro, como no modelo anterior
arquivo = st.file_uploader(
    "Carregue uma foto de documento",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    key="upload_imagem",
)

# Sidebar com o estilo antigo: controles visíveis + botão de reset
st.sidebar.header("Ajustes do processamento")

if st.sidebar.button("Resetar filtros"):
    resetar_filtros()
    st.rerun()

st.sidebar.selectbox(
    "Resultado final",
    ["Automático inteligente", "Colorido recortado", "Cinza aprimorado", "Scanner P&B"],
    key="modo_saida",
)

st.sidebar.checkbox(
    "Detectar e recortar documento automaticamente",
    key="usar_deteccao",
)

st.sidebar.slider("Canny mínimo", 0, 255, key="canny_min")
st.sidebar.slider("Canny máximo", 0, 255, key="canny_max")
st.sidebar.slider("Brilho", -50, 80, key="brilho")
st.sidebar.slider("Contraste", 0.6, 2.0, step=0.05, key="contraste")

st.sidebar.selectbox(
    "Método de suavização",
    ["Gaussiano", "Mediana", "Bilateral", "Nenhum"],
    key="metodo_suavizacao",
)

st.sidebar.slider("Força da suavização", 3, 15, step=2, key="kernel_suavizacao")
st.sidebar.slider("Força do CLAHE", 0.5, 5.0, step=0.1, key="forca_clahe")
st.sidebar.slider("Grade do CLAHE", 4, 16, step=1, key="grid_clahe")
st.sidebar.slider("Bloco adaptativo", 3, 99, step=2, key="bloco")
st.sidebar.slider("Constante adaptativa", 0, 30, key="constante")
st.sidebar.checkbox("Limpeza morfológica", key="limpeza_morfologica")

if arquivo is None:
    st.info("Envie uma imagem para iniciar o processamento.")
    st.stop()

imagem_pil = Image.open(arquivo).convert("RGB")
imagem_rgb = np.array(imagem_pil)

etapas, deteccao, modo_final = processar_documento(
    imagem_rgb,
    usar_deteccao=st.session_state.usar_deteccao,
    modo_saida=st.session_state.modo_saida,
    canny_min=st.session_state.canny_min,
    canny_max=st.session_state.canny_max,
    brilho=st.session_state.brilho,
    contraste=st.session_state.contraste,
    metodo_suavizacao=st.session_state.metodo_suavizacao,
    kernel_suavizacao=st.session_state.kernel_suavizacao,
    forca_clahe=st.session_state.forca_clahe,
    grid_clahe=st.session_state.grid_clahe,
    bloco=st.session_state.bloco,
    constante=st.session_state.constante,
    limpeza_morfologica=st.session_state.limpeza_morfologica,
)

if st.session_state.usar_deteccao:
    if deteccao["detectado"]:
        st.success("Documento detectado automaticamente. Perspectiva corrigida.")
    else:
        st.warning(
            "Não foi possível detectar automaticamente as bordas do documento. "
            "A ferramenta usou a imagem inteira como fallback."
        )
else:
    st.info("Detecção automática desligada. A ferramenta está processando a imagem inteira.")

# Comparação principal minimizada, mas ainda legível
st.subheader("Comparação principal")
col1, col2, col3 = st.columns(3)

with col1:
    st.image(etapas["01_original"], caption="Imagem original", width=330)
    st.download_button(
        "Baixar original",
        data=imagem_para_png(etapas["01_original"]),
        file_name="01_original.png",
        mime="image/png",
    )

with col2:
    st.image(etapas["04_documento_recortado_colorido"], caption="Documento detectado/recortado", width=330)
    st.download_button(
        "Baixar recorte",
        data=imagem_para_png(etapas["04_documento_recortado_colorido"]),
        file_name="04_documento_recortado_colorido.png",
        mime="image/png",
    )

with col3:
    st.image(etapas["10_resultado_final"], caption=f"Resultado final - {modo_final}", width=330)
    st.download_button(
        "Baixar resultado final PNG",
        data=imagem_para_png(etapas["10_resultado_final"]),
        file_name="docscan_resultado_final.png",
        mime="image/png",
    )
    st.download_button(
        "Baixar resultado final PDF",
        data=imagem_para_pdf(etapas["10_resultado_final"]),
        file_name="docscan_resultado_final.pdf",
        mime="application/pdf",
    )

st.markdown("---")

st.subheader("Download geral")
st.download_button(
    "Baixar ZIP com todas as imagens",
    data=gerar_zip(etapas),
    file_name="docscan_pi_resultados.zip",
    mime="application/zip",
)

st.markdown("---")

st.subheader("Etapas do processamento")

titulos = {
    "02_bordas_para_detectar": "1. Canny para detectar documento",
    "03_bordas_dilatadas": "2. Bordas dilatadas",
    "04_documento_recortado_colorido": "3. Documento recortado / perspectiva",
    "05_escala_cinza": "4. Escala de cinza",
    "06_correcao_iluminacao": "5. Correção de iluminação",
    "07_equalizacao_histograma": "6. Equalização de histograma",
    "08_contraste_clahe_brilho": "7. CLAHE + brilho/contraste",
    "09_suavizacao": "8. Suavização",
    "11_scanner_preto_e_branco": "9. Scanner P&B suave",
    "12_limiarizacao_otsu": "10. Limiarização de Otsu",
    "13_limiarizacao_adaptativa": "11. Limiarização adaptativa",
    "14_bordas_canny_processamento": "12. Canny do processamento",
    "15_threshold_global": "13. Threshold global",
}

itens = list(titulos.items())

for i in range(0, len(itens), 3):
    col_a, col_b, col_c = st.columns(3)
    for coluna, item in zip([col_a, col_b, col_c], itens[i:i + 3]):
        nome, titulo = item
        with coluna:
            st.markdown(f"**{titulo}**")
            st.image(etapas[nome], width=260)
            st.download_button(
                "Baixar PNG",
                data=imagem_para_png(etapas[nome]),
                file_name=f"{nome}.png",
                mime="image/png",
                key=f"download_{nome}",
            )

st.success("Processamento concluído com sucesso.")

st.markdown("---")
st.markdown(
    """
### Fluxo aplicado

Imagem original → detecção de bordas → contornos → maior quadrilátero → correção de perspectiva →
escala de cinza → correção de iluminação → equalização/CLAHE → brilho e contraste → suavização →
limiarização → resultado final.
"""
)
