from pathlib import Path
import argparse

import cv2 # type: ignore
import numpy as np


def carregar_imagem(caminho):
    imagem = cv2.imread(caminho)

    if imagem is None:
        raise FileNotFoundError(f"Nao foi possivel carregar a imagem: {caminho}")

    return imagem


def salvar_imagem(pasta_saida, nome, imagem):
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho = pasta_saida / nome
    cv2.imwrite(str(caminho), imagem)
    return caminho


def converter_para_cinza(imagem):
    """
    Converte a imagem colorida para escala de cinza.
    Esse passo facilita a aplicacao de filtros de borda e realce.
    """
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def filtro_media(imagem, tamanho_kernel=5):
    """
    Filtro da media.
    Suaviza a imagem substituindo cada pixel pela media dos pixels vizinhos.
    Reduz ruido, mas pode borrar bordas.
    """
    return cv2.blur(imagem, (tamanho_kernel, tamanho_kernel))


def filtro_mediana(imagem, tamanho_kernel=5):
    """
    Filtro da mediana.
    Substitui cada pixel pela mediana da vizinhanca.
    Muito usado para remover ruido do tipo sal e pimenta.
    """
    return cv2.medianBlur(imagem, tamanho_kernel)


def filtro_gaussiano(imagem, tamanho_kernel=5):
    """
    Filtro Gaussiano.
    Suaviza a imagem usando uma mascara com pesos gaussianos.
    Reduz ruido de forma mais natural que o filtro da media.
    """
    return cv2.GaussianBlur(imagem, (tamanho_kernel, tamanho_kernel), 0)


def filtro_bilateral(imagem):
    """
    Filtro bilateral.
    Suaviza a imagem preservando bordas.
    Considera tanto a distancia espacial quanto a diferenca de intensidade.
    """
    return cv2.bilateralFilter(imagem, 9, 75, 75)


def filtro_sobel(imagem):
    """
    Filtro Sobel.
    Detecta bordas calculando gradientes nas direcoes X e Y.
    """
    cinza = converter_para_cinza(imagem)

    sobel_x = cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(cinza, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(sobel_x, sobel_y)

    return cv2.convertScaleAbs(magnitude)


def filtro_prewitt(imagem):
    """
    Filtro Prewitt.
    Detecta bordas usando mascaras de gradiente horizontal e vertical.
    """
    cinza = converter_para_cinza(imagem)

    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    kernel_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])

    prewitt_x = cv2.filter2D(cinza, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(cinza, cv2.CV_64F, kernel_y)

    magnitude = cv2.magnitude(prewitt_x, prewitt_y)

    return cv2.convertScaleAbs(magnitude)


def filtro_laplaciano(imagem):
    """
    Filtro Laplaciano.
    Realca regioes com mudanca brusca de intensidade.
    Detecta bordas em varias direcoes.
    """
    cinza = converter_para_cinza(imagem)
    laplaciano = cv2.Laplacian(cinza, cv2.CV_64F)

    return cv2.convertScaleAbs(laplaciano)


def filtro_canny(imagem):
    """
    Filtro Canny.
    Detecta bordas usando suavizacao, gradiente e limiares.
    E um dos metodos mais conhecidos para deteccao de bordas.
    """
    cinza = converter_para_cinza(imagem)
    suavizada = filtro_gaussiano(cinza)

    return cv2.Canny(suavizada, 80, 160)


def filtro_nitidez(imagem):
    """
    Filtro de nitidez.
    Usa uma mascara que reforca o pixel central e destaca detalhes.
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(imagem, -1, kernel)


def filtro_passa_alta(imagem):
    """
    Filtro passa-alta.
    Realca detalhes e mudancas bruscas de intensidade.
    """
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    return cv2.filter2D(imagem, -1, kernel)


def filtro_passa_baixa(imagem):
    """
    Filtro passa-baixa.
    Suaviza a imagem e reduz altas frequencias, como ruido.
    """
    kernel = np.ones((5, 5), np.float32) / 25

    return cv2.filter2D(imagem, -1, kernel)


def erosao(imagem):
    """
    Erosao.
    Reduz regioes claras da imagem.
    Pode remover pequenos ruidos brancos.
    """
    kernel = np.ones((3, 3), np.uint8)

    return cv2.erode(imagem, kernel, iterations=1)


def dilatacao(imagem):
    """
    Dilatacao.
    Expande regioes claras da imagem.
    Pode preencher pequenas falhas.
    """
    kernel = np.ones((3, 3), np.uint8)

    return cv2.dilate(imagem, kernel, iterations=1)


def abertura(imagem):
    """
    Abertura.
    Erosao seguida de dilatacao.
    Remove pequenos ruidos claros.
    """
    kernel = np.ones((5, 5), np.uint8)

    return cv2.morphologyEx(imagem, cv2.MORPH_OPEN, kernel)


def fechamento(imagem):
    """
    Fechamento.
    Dilatacao seguida de erosao.
    Preenche pequenas falhas em regioes claras.
    """
    kernel = np.ones((5, 5), np.uint8)

    return cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, kernel)


def aplicar_filtros(caminho_imagem, pasta_saida):
    imagem = carregar_imagem(caminho_imagem)

    resultados = {
        "01_original.png": imagem,
        "02_escala_cinza.png": converter_para_cinza(imagem),
        "03_filtro_media.png": filtro_media(imagem),
        "04_filtro_mediana.png": filtro_mediana(imagem),
        "05_filtro_gaussiano.png": filtro_gaussiano(imagem),
        "06_filtro_bilateral.png": filtro_bilateral(imagem),
        "07_filtro_sobel.png": filtro_sobel(imagem),
        "08_filtro_prewitt.png": filtro_prewitt(imagem),
        "09_filtro_laplaciano.png": filtro_laplaciano(imagem),
        "10_filtro_canny.png": filtro_canny(imagem),
        "11_filtro_nitidez.png": filtro_nitidez(imagem),
        "12_filtro_passa_alta.png": filtro_passa_alta(imagem),
        "13_filtro_passa_baixa.png": filtro_passa_baixa(imagem),
        "14_erosao.png": erosao(imagem),
        "15_dilatacao.png": dilatacao(imagem),
        "16_abertura.png": abertura(imagem),
        "17_fechamento.png": fechamento(imagem),
    }

    caminhos_salvos = []

    for nome, resultado in resultados.items():
        caminho = salvar_imagem(pasta_saida, nome, resultado)
        caminhos_salvos.append(caminho)

    return caminhos_salvos


def main():
    parser = argparse.ArgumentParser(
        description="Aplicacao de filtros de Processamento de Imagens Digitais"
    )

    parser.add_argument(
        "imagem",
        help="Caminho da imagem de entrada"
    )

    parser.add_argument(
        "--saida",
        default="resultados_filtros",
        help="Pasta onde os resultados serao salvos"
    )

    args = parser.parse_args()

    pasta_saida = Path(args.saida)
    caminhos = aplicar_filtros(args.imagem, pasta_saida)

    print("\nFiltros aplicados com sucesso!\n")
    print("Arquivos gerados:")

    for caminho in caminhos:
        print(f"- {caminho}")


if __name__ == "__main__":
    main()