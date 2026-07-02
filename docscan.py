import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import zipfile


# ============================================================
# CONFIGURAÇÕES PADRÃO
# ============================================================

VALORES_PADRAO = {
    "brilho": 10,
    "contraste": 1.15,
    "canny_limiar1": 50,
    "canny_limiar2": 150,
    "bloco": 31,
    "constante": 11,
    "clahe_clip": 2.0,
    "margem_canvas": 30,
}


def inicializar_sliders():
    for chave, valor in VALORES_PADRAO.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


def resetar_sliders():
    for chave, valor in VALORES_PADRAO.items():
        st.session_state[chave] = valor


# ============================================================
# FUNÇÕES BÁSICAS
# ============================================================

def garantir_impar(valor, minimo=3):
    valor = int(valor)
    if valor < minimo:
        valor = minimo
    if valor % 2 == 0:
        valor += 1
    return valor


def converter_para_cinza(imagem_rgb):
    return cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)


def ajustar_brilho_contraste(imagem, brilho=10, contraste=1.15):
    return cv2.convertScaleAbs(imagem, alpha=float(contraste), beta=int(brilho))


def aplicar_suavizacao(imagem, metodo="Gaussiano"):
    if metodo == "Mediana":
        return cv2.medianBlur(imagem, 5)

    if metodo == "Bilateral":
        return cv2.bilateralFilter(imagem, 9, 75, 75)

    return cv2.GaussianBlur(imagem, (5, 5), 0)


def imagem_para_download(imagem):
    buffer = BytesIO()

    if len(imagem.shape) == 2:
        imagem_pil = Image.fromarray(imagem)
    else:
        imagem_pil = Image.fromarray(imagem)

    imagem_pil.save(buffer, format="PNG")
    return buffer.getvalue()


def gerar_pdf_da_imagem(imagem):
    buffer = BytesIO()

    if len(imagem.shape) == 2:
        imagem_pil = Image.fromarray(imagem).convert("RGB")
    else:
        imagem_pil = Image.fromarray(imagem).convert("RGB")

    imagem_pil.save(buffer, format="PDF", resolution=100.0)
    return buffer.getvalue()


def gerar_zip_com_imagens(etapas):
    buffer_zip = BytesIO()

    with zipfile.ZipFile(buffer_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for nome, imagem in etapas.items():
            zip_file.writestr(f"{nome}.png", imagem_para_download(imagem))

        zip_file.writestr(
            "resultado_final.pdf",
            gerar_pdf_da_imagem(etapas["10_resultado_final"])
        )

    buffer_zip.seek(0)
    return buffer_zip.getvalue()


# ============================================================
# DETECÇÃO, PERSPECTIVA, ALINHAMENTO E RECORTE
# ============================================================

def ordenar_pontos(pontos):
    pontos = np.asarray(pontos, dtype="float32").reshape(4, 2)

    retangulo = np.zeros((4, 2), dtype="float32")
    soma = pontos.sum(axis=1)
    diferenca = np.diff(pontos, axis=1)

    retangulo[0] = pontos[np.argmin(soma)]
    retangulo[2] = pontos[np.argmax(soma)]
    retangulo[1] = pontos[np.argmin(diferenca)]
    retangulo[3] = pontos[np.argmax(diferenca)]

    return retangulo


def calcular_angulo(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    coseno = np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8)
    coseno = np.clip(coseno, -1.0, 1.0)

    return np.degrees(np.arccos(coseno))


def validar_quadrilatero(pontos, area_imagem):
    pontos = np.asarray(pontos, dtype="float32").reshape(-1, 2)

    if len(pontos) != 4:
        return False

    contorno = pontos.astype(np.int32).reshape(-1, 1, 2)

    if not cv2.isContourConvex(contorno):
        return False

    area = cv2.contourArea(contorno)
    proporcao_area = area / float(area_imagem)

    if proporcao_area < 0.08 or proporcao_area > 0.98:
        return False

    rect = ordenar_pontos(pontos)
    tl, tr, br, bl = rect

    largura_a = np.linalg.norm(br - bl)
    largura_b = np.linalg.norm(tr - tl)
    altura_a = np.linalg.norm(tr - br)
    altura_b = np.linalg.norm(tl - bl)

    largura = max(largura_a, largura_b)
    altura = max(altura_a, altura_b)

    if largura < 80 or altura < 80:
        return False

    proporcao = max(largura, altura) / (min(largura, altura) + 1e-8)

    if proporcao > 8.0:
        return False

    angulos = [
        calcular_angulo(bl, tl, tr),
        calcular_angulo(tl, tr, br),
        calcular_angulo(tr, br, bl),
        calcular_angulo(br, bl, tl),
    ]

    for angulo in angulos:
        if angulo < 45 or angulo > 135:
            return False

    return True


def transformar_perspectiva(imagem, pontos):
    rect = ordenar_pontos(pontos)
    tl, tr, br, bl = rect

    largura_a = np.linalg.norm(br - bl)
    largura_b = np.linalg.norm(tr - tl)
    largura = int(max(largura_a, largura_b))

    altura_a = np.linalg.norm(tr - br)
    altura_b = np.linalg.norm(tl - bl)
    altura = int(max(altura_a, altura_b))

    largura = max(largura, 100)
    altura = max(altura, 100)

    destino = np.array(
        [
            [0, 0],
            [largura - 1, 0],
            [largura - 1, altura - 1],
            [0, altura - 1],
        ],
        dtype="float32"
    )

    matriz = cv2.getPerspectiveTransform(rect, destino)
    corrigida = cv2.warpPerspective(imagem, matriz, (largura, altura))

    return corrigida


def rotacionar_imagem(imagem, angulo):
    altura, largura = imagem.shape[:2]
    centro = (largura // 2, altura // 2)

    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)

    cos = abs(matriz[0, 0])
    sin = abs(matriz[0, 1])

    nova_largura = int((altura * sin) + (largura * cos))
    nova_altura = int((altura * cos) + (largura * sin))

    matriz[0, 2] += (nova_largura / 2) - centro[0]
    matriz[1, 2] += (nova_altura / 2) - centro[1]

    return cv2.warpAffine(
        imagem,
        matriz,
        (nova_largura, nova_altura),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def corrigir_rotacao_documento(imagem):
    cinza = converter_para_cinza(imagem) if len(imagem.shape) == 3 else imagem.copy()
    blur = cv2.GaussianBlur(cinza, (3, 3), 0)

    _, binaria = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    pontos = cv2.findNonZero(binaria)

    if pontos is None or len(pontos) < 80:
        return imagem

    retangulo = cv2.minAreaRect(pontos)
    angulo = retangulo[-1]

    if angulo < -45:
        angulo = -(90 + angulo)
    else:
        angulo = -angulo

    if abs(angulo) < 0.3 or abs(angulo) > 12:
        return imagem

    return rotacionar_imagem(imagem, angulo)


def recorte_fino_documento(imagem):
    cinza = converter_para_cinza(imagem) if len(imagem.shape) == 3 else imagem.copy()

    blur = cv2.GaussianBlur(cinza, (5, 5), 0)

    _, mascara_texto = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara_texto, cv2.MORPH_CLOSE, kernel, iterations=2)

    pontos = cv2.findNonZero(mascara)

    if pontos is None:
        return imagem

    x, y, w, h = cv2.boundingRect(pontos)

    area_crop = w * h
    area_total = imagem.shape[0] * imagem.shape[1]

    if area_crop < area_total * 0.05:
        return imagem

    margem = int(max(12, min(imagem.shape[:2]) * 0.04))

    x1 = max(x - margem, 0)
    y1 = max(y - margem, 0)
    x2 = min(x + w + margem, imagem.shape[1])
    y2 = min(y + h + margem, imagem.shape[0])

    if x2 <= x1 or y2 <= y1:
        return imagem

    return imagem[y1:y2, x1:x2]


def centralizar_em_canvas(imagem, margem=30, cor_fundo=255):
    margem = int(margem)

    if len(imagem.shape) == 2:
        altura, largura = imagem.shape
        canvas = np.full(
            (altura + margem * 2, largura + margem * 2),
            cor_fundo,
            dtype=np.uint8
        )
        canvas[margem:margem + altura, margem:margem + largura] = imagem
        return canvas

    altura, largura = imagem.shape[:2]
    canvas = np.full(
        (altura + margem * 2, largura + margem * 2, 3),
        cor_fundo,
        dtype=np.uint8
    )
    canvas[margem:margem + altura, margem:margem + largura] = imagem

    return canvas


def detectar_por_contorno_quadrilateral(imagem):
    altura_original = imagem.shape[0]
    escala = altura_original / 800.0

    largura_redimensionada = int(imagem.shape[1] / escala)
    pequena = cv2.resize(imagem, (largura_redimensionada, 800))

    cinza = converter_para_cinza(pequena)
    suavizada = cv2.bilateralFilter(cinza, 9, 75, 75)

    bordas = cv2.Canny(
        suavizada,
        st.session_state.canny_limiar1,
        st.session_state.canny_limiar2
    )

    kernel = np.ones((5, 5), np.uint8)
    bordas_unidas = cv2.dilate(bordas, kernel, iterations=1)
    bordas_unidas = cv2.morphologyEx(bordas_unidas, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(
        bordas_unidas,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    area_imagem = pequena.shape[0] * pequena.shape[1]

    desenho = imagem.copy()

    for contorno in contornos[:20]:
        perimetro = cv2.arcLength(contorno, True)

        for fator in [0.015, 0.02, 0.03, 0.04]:
            aproximado = cv2.approxPolyDP(contorno, fator * perimetro, True)

            if len(aproximado) != 4:
                continue

            pontos_pequenos = aproximado.reshape(4, 2)

            if not validar_quadrilatero(pontos_pequenos, area_imagem):
                continue

            pontos_originais = pontos_pequenos.astype("float32") * escala

            cv2.drawContours(
                desenho,
                [pontos_originais.astype(np.int32).reshape(-1, 1, 2)],
                -1,
                (0, 255, 0),
                8
            )

            perspectiva = transformar_perspectiva(imagem, pontos_originais)

            return {
                "sucesso": True,
                "documento": perspectiva,
                "bordas": bordas,
                "contorno_desenhado": desenho,
                "metodo": "Contorno quadrilateral"
            }

    return {
        "sucesso": False,
        "documento": imagem,
        "bordas": bordas,
        "contorno_desenhado": desenho,
        "metodo": "Falhou"
    }


def detectar_por_regiao_clara(imagem):
    cinza = converter_para_cinza(imagem)

    blur = cv2.GaussianBlur(cinza, (5, 5), 0)

    _, mascara = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((9, 9), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return {
            "sucesso": False,
            "documento": imagem,
            "contorno_desenhado": imagem.copy(),
            "metodo": "Fallback falhou"
        }

    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    area_total = imagem.shape[0] * imagem.shape[1]

    desenho = imagem.copy()

    for contorno in contornos[:10]:
        area = cv2.contourArea(contorno)

        if area < area_total * 0.08 or area > area_total * 0.98:
            continue

        x, y, w, h = cv2.boundingRect(contorno)

        if w < 80 or h < 80:
            continue

        margem = int(max(15, min(w, h) * 0.04))

        x1 = max(x - margem, 0)
        y1 = max(y - margem, 0)
        x2 = min(x + w + margem, imagem.shape[1])
        y2 = min(y + h + margem, imagem.shape[0])

        cv2.rectangle(desenho, (x1, y1), (x2, y2), (255, 180, 0), 8)

        recorte = imagem[y1:y2, x1:x2]

        return {
            "sucesso": True,
            "documento": recorte,
            "contorno_desenhado": desenho,
            "metodo": "Região clara"
        }

    return {
        "sucesso": False,
        "documento": imagem,
        "contorno_desenhado": desenho,
        "metodo": "Fallback falhou"
    }


def detectar_e_alinhar_documento(imagem):
    resultado = detectar_por_contorno_quadrilateral(imagem)

    if resultado["sucesso"]:
        documento = resultado["documento"]
        detectado = True
        metodo = resultado["metodo"]
        bordas = resultado["bordas"]
        contorno_desenhado = resultado["contorno_desenhado"]
    else:
        fallback = detectar_por_regiao_clara(imagem)
        documento = fallback["documento"]
        detectado = fallback["sucesso"]
        metodo = fallback["metodo"]
        bordas = resultado["bordas"]
        contorno_desenhado = fallback["contorno_desenhado"]

    rotacionado = corrigir_rotacao_documento(documento)
    recortado = recorte_fino_documento(rotacionado)
    centralizado = centralizar_em_canvas(recortado, margem=st.session_state.margem_canvas)

    return {
        "detectado": detectado,
        "metodo": metodo,
        "bordas": bordas,
        "contorno_desenhado": contorno_desenhado,
        "perspectiva": documento,
        "rotacionado": rotacionado,
        "recortado": recortado,
        "centralizado": centralizado
    }


# ============================================================
# FILTROS FINAIS
# ============================================================

def corrigir_iluminacao(imagem_cinza):
    fundo = cv2.GaussianBlur(imagem_cinza, (0, 0), sigmaX=35, sigmaY=35)
    corrigida = cv2.divide(imagem_cinza, fundo, scale=255)
    corrigida = cv2.normalize(corrigida, None, 0, 255, cv2.NORM_MINMAX)
    return corrigida.astype(np.uint8)


def gerar_cinza_aprimorado(documento):
    cinza = converter_para_cinza(documento) if len(documento.shape) == 3 else documento.copy()

    sem_sombra = corrigir_iluminacao(cinza)

    clahe = cv2.createCLAHE(
        clipLimit=float(st.session_state.clahe_clip),
        tileGridSize=(8, 8)
    )

    contraste_local = clahe.apply(sem_sombra)

    ajustada = ajustar_brilho_contraste(
        contraste_local,
        brilho=st.session_state.brilho,
        contraste=st.session_state.contraste
    )

    suavizada = aplicar_suavizacao(
        ajustada,
        metodo=st.session_state.metodo_suavizacao
    )

    return suavizada


def gerar_scanner_pb(imagem_cinza):
    bloco = garantir_impar(st.session_state.bloco, minimo=3)

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
        st.session_state.constante
    )

    combinada = cv2.bitwise_and(otsu, adaptativa)

    kernel = np.ones((2, 2), np.uint8)
    limpa = cv2.morphologyEx(combinada, cv2.MORPH_OPEN, kernel)

    return limpa


def escolher_resultado_final(documento, cinza_aprimorado, scanner_pb):
    if st.session_state.modo_saida == "Cinza aprimorado":
        return cinza_aprimorado

    if st.session_state.modo_saida == "Scanner P&B":
        return scanner_pb

    if st.session_state.modo_saida == "Colorido centralizado":
        return documento

    if len(documento.shape) == 3:
        hsv = cv2.cvtColor(documento, cv2.COLOR_RGB2HSV)
        saturacao_media = hsv[:, :, 1].mean()
    else:
        saturacao_media = 0

    proporcao_preto = np.mean(scanner_pb < 128)

    documento_colorido = saturacao_media > 35
    pb_saudavel = 0.03 <= proporcao_preto <= 0.45

    if documento_colorido:
        return cinza_aprimorado

    if pb_saudavel:
        return scanner_pb

    return cinza_aprimorado


def gerar_resultado_final(documento_alinhado, filtros=None):
    cinza_aprimorado = gerar_cinza_aprimorado(documento_alinhado)
    scanner_pb = gerar_scanner_pb(cinza_aprimorado)
    final = escolher_resultado_final(documento_alinhado, cinza_aprimorado, scanner_pb)

    return {
        "cinza_aprimorado": cinza_aprimorado,
        "scanner_pb": scanner_pb,
        "final": final
    }


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.set_page_config(
    page_title="DOCSCAN PI",
    page_icon="📄",
    layout="wide"
)

inicializar_sliders()

st.title("📄 DOCSCAN PI")
st.write(
    "Scanner automático de documentos com detecção de bordas, correção de perspectiva, "
    "alinhamento, recorte fino e filtros de legibilidade."
)

st.sidebar.header("Controles")

if st.sidebar.button("Resetar filtros"):
    resetar_sliders()
    st.rerun()

st.sidebar.selectbox(
    "Modo de saída",
    ["Automático inteligente", "Colorido centralizado", "Cinza aprimorado", "Scanner P&B"],
    key="modo_saida"
)

st.sidebar.selectbox(
    "Método de suavização",
    ["Gaussiano", "Mediana", "Bilateral"],
    key="metodo_suavizacao"
)

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
    step=0.05,
    key="contraste"
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
    "CLAHE - Clip limit",
    min_value=1.0,
    max_value=5.0,
    step=0.1,
    key="clahe_clip"
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

st.sidebar.slider(
    "Margem do canvas",
    min_value=0,
    max_value=100,
    key="margem_canvas"
)

arquivo = st.file_uploader(
    "Carregue uma imagem de documento",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if arquivo is None:
    st.info("Envie uma imagem para iniciar o processamento.")
    st.stop()

imagem_pil = Image.open(arquivo).convert("RGB")
imagem_rgb = np.array(imagem_pil)

alinhamento = detectar_e_alinhar_documento(imagem_rgb)
resultado = gerar_resultado_final(alinhamento["centralizado"])

if alinhamento["detectado"]:
    st.success(f"Documento detectado automaticamente. Método: {alinhamento['metodo']}.")
else:
    st.warning(
        "Não foi possível detectar automaticamente um documento confiável. "
        "O app usou a imagem original como fallback."
    )

st.subheader("Comparação antes e depois")

col1, col2 = st.columns(2)

with col1:
    st.image(imagem_rgb, caption="Imagem original", use_container_width=True)

with col2:
    st.image(resultado["final"], caption="Resultado final", use_container_width=True)

etapas = {
    "01_imagem_original": imagem_rgb,
    "02_bordas_canny": alinhamento["bordas"],
    "03_contorno_detectado": alinhamento["contorno_desenhado"],
    "04_perspectiva_corrigida": alinhamento["perspectiva"],
    "05_documento_rotacionado": alinhamento["rotacionado"],
    "06_recorte_fino": alinhamento["recortado"],
    "07_canvas_centralizado": alinhamento["centralizado"],
    "08_cinza_aprimorado": resultado["cinza_aprimorado"],
    "09_scanner_pb": resultado["scanner_pb"],
    "10_resultado_final": resultado["final"],
}

st.markdown("---")
st.subheader("Etapas do pipeline")

nomes_legiveis = {
    "01_imagem_original": "1. Imagem original",
    "02_bordas_canny": "2. Bordas Canny",
    "03_contorno_detectado": "3. Contorno detectado",
    "04_perspectiva_corrigida": "4. Perspectiva corrigida",
    "05_documento_rotacionado": "5. Documento alinhado",
    "06_recorte_fino": "6. Recorte fino",
    "07_canvas_centralizado": "7. Canvas centralizado",
    "08_cinza_aprimorado": "8. Cinza aprimorado",
    "09_scanner_pb": "9. Scanner P&B",
    "10_resultado_final": "10. Resultado final",
}

itens = list(etapas.items())

for i in range(0, len(itens), 3):
    col_a, col_b, col_c = st.columns(3)

    for coluna, item in zip([col_a, col_b, col_c], itens[i:i + 3]):
        nome, imagem = item

        with coluna:
            st.markdown(f"**{nomes_legiveis[nome]}**")
            st.image(imagem, use_container_width=True)

            st.download_button(
                label="Baixar PNG",
                data=imagem_para_download(imagem),
                file_name=f"{nome}.png",
                mime="image/png",
                key=f"download_{nome}"
            )

st.markdown("---")
st.subheader("Downloads")

col_png, col_pdf, col_zip = st.columns(3)

with col_png:
    st.download_button(
        label="Baixar resultado final PNG",
        data=imagem_para_download(etapas["10_resultado_final"]),
        file_name="resultado_final_docscan.png",
        mime="image/png"
    )

with col_pdf:
    st.download_button(
        label="Baixar resultado final PDF",
        data=gerar_pdf_da_imagem(etapas["10_resultado_final"]),
        file_name="resultado_final_docscan.pdf",
        mime="application/pdf"
    )

with col_zip:
    st.download_button(
        label="Baixar ZIP com todas as imagens",
        data=gerar_zip_com_imagens(etapas),
        file_name="docscan_pi_resultados.zip",
        mime="application/zip"
    )

st.success("Processamento concluído.")
