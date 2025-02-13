import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms
from train import CNN  # Certifique-se de que train.py contém sua CNN
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance

# Carregar o modelo treinado
model = CNN()
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

# Transformações de imagem
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Função para previsão de probabilidades
def predict_probabilities(img):
    img_tensor = transform(img).unsqueeze(0)
    output = model(img_tensor)
    probabilities = F.softmax(output, dim=1)
    return probabilities.squeeze().tolist()

# Interface Streamlit
st.title("Reconhecedor de dígito")

# Layout de duas colunas
col1, col2 = st.columns(2)

# Canvas interativo na coluna da esquerda
with col1:
    st.subheader("Desenhe um dígito")
    canvas_result = st_canvas(
        #fill_color="black",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# Processar a imagem do canvas e mostrar as probabilidades na coluna da direita
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0] * 255).astype("uint8"))
    img = img.resize((28, 28)).convert("L")
    #img = ImageOps.invert(img)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(100.0)

    with col1:
        # Mostrar a imagem processada abaixo do canvas
        st.image(img, caption="Processed Image", width=280)

    with col2:
        # Fazer previsão e mostrar as probabilidades
        st.subheader("Predição")
        probabilities = predict_probabilities(img)

        for i, prob in enumerate(probabilities):
            st.write(f"{i}: {prob:.2%}")
            st.progress(int(prob * 100))
else:
    with col2:
        # Fazer previsão e mostrar as probabilidades
        st.subheader("Predição")

        for i in range(10):
            st.write(f"{i}: {0:.2%}")