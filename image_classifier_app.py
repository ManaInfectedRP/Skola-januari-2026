import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess, decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import pandas as pd
from datetime import datetime

# Konfigurera sidan
st.set_page_config(
    page_title="Bildklassificering med AI",
    page_icon="",
    layout="wide"
)

# Titel och beskrivning
st.title("AI Bildklassificerare")
st.write("""
Denna applikation använder förtränade CNN-modeller för att klassificera bilder.
""")

# Model configurations
MODEL_INFO = {
    'ResNet50': {
        'model_class': ResNet50,
        'preprocess': resnet_preprocess,
        'target_size': (224, 224),
        'description': 'Djup residual network med 50 lager. Bra generell prestanda.'
    },
    'VGG16': {
        'model_class': VGG16,
        'preprocess': vgg_preprocess,
        'target_size': (224, 224),
        'description': 'Klassisk arkitektur med 16 lager. Enkel men effektiv.'
    },
    'MobileNetV2': {
        'model_class': MobileNetV2,
        'preprocess': mobilenet_preprocess,
        'target_size': (224, 224),
        'description': 'Lätt och snabb modell, optimerad för mobila enheter.'
    },
    'InceptionV3': {
        'model_class': InceptionV3,
        'preprocess': inception_preprocess,
        'target_size': (299, 299),
        'description': 'Använder inception-moduler för multi-scale features.'
    }
}

# Ladda modell (cachad för bättre prestanda)
@st.cache_resource
def load_model(model_name):
    model_class = MODEL_INFO[model_name]['model_class']
    return model_class(weights='imagenet')

# Funktion för att applicera bildfilter
def apply_image_filters(img, brightness, contrast, sharpness):
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
    
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)
    
    return img

# Funktion för att prediktera bild
def predict_image(img, model, model_name, top_k=5):
    # Hämta target size och preprocess funktion för vald modell
    target_size = MODEL_INFO[model_name]['target_size']
    preprocess_fn = MODEL_INFO[model_name]['preprocess']
    
    # Resize bilden
    img_resized = img.resize(target_size)

    # Konvertera till array
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_fn(x)

    # Prediktera
    preds = model.predict(x, verbose=0)
    return decode_predictions(preds, top=top_k)[0]

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Sidebar för inställningar
st.sidebar.header("Inställningar")

# Model selection
selected_model = st.sidebar.selectbox(
    "Välj AI Modell:",
    options=list(MODEL_INFO.keys()),
    help="Olika modeller har olika styrkor och svagheter"
)

st.sidebar.caption(MODEL_INFO[selected_model]['description'])

# Ladda modell
with st.spinner(f'Laddar {selected_model}-modell...'):
    model = load_model(selected_model)
st.sidebar.success('Modell laddad!')

# Antal prediktioner att visa
top_k = st.sidebar.slider(
    "Antal prediktioner:",
    min_value=1,
    max_value=10,
    value=5,
    help="Hur många top-prediktioner som ska visas"
)

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Visa prediktioner över:",
    min_value=0.0,
    max_value=1.0,
    value=0.01,
    step=0.01,
    format="%.2f"
)

st.sidebar.markdown("---")
st.sidebar.header("Bildfilter")

apply_filters = st.sidebar.checkbox("Aktivera bildfilter", value=False)

if apply_filters:
    brightness = st.sidebar.slider("Ljusstyrka:", 0.5, 2.0, 1.0, 0.1)
    contrast = st.sidebar.slider("Kontrast:", 0.5, 2.0, 1.0, 0.1)
    sharpness = st.sidebar.slider("Skärpa:", 0.5, 2.0, 1.0, 0.1)
else:
    brightness = contrast = sharpness = 1.0

# Tabs för olika funktioner
tab1, tab2, tab3 = st.tabs(["📤 Ladda upp bild", "📷 Använd kamera", "📊 Historik"])

with tab1:
    st.header("📤 Ladda upp en bild")
    
    # Filuppladdare med multi-upload
    uploaded_files = st.file_uploader(
        "Välj en eller flera bilder...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Ladda upp bilder i JPG, JPEG eller PNG format"
    )

    if uploaded_files:
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"### Bild {idx + 1}: {uploaded_file.name}")
            
            # Läs bilden
            img = Image.open(uploaded_file)
            original_img = img.copy()
            
            # Applicera filter om aktiverade
            if apply_filters:
                img = apply_image_filters(img, brightness, contrast, sharpness)

            # Visa bild och prediktioner sida vid sida
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Bild")
                st.image(img, use_container_width=True)
                st.caption(f"Storlek: {original_img.size[0]}x{original_img.size[1]} pixels")

            with col2:
                st.subheader("AI Prediktioner")

                with st.spinner('Analyserar bild...'):
                    predictions = predict_image(img, model, selected_model, top_k)

                st.write("### Resultat:")

                # Filtrera prediktioner baserat på threshold
                filtered_preds = [(name, prob) for _, name, prob in predictions if prob >= confidence_threshold]

                if filtered_preds:
                    for i, (class_name, probability) in enumerate(filtered_preds, 1):
                        # Formatera klassnamn
                        display_name = class_name.replace('_', ' ').title()

                        # Progress bar för sannolikhet
                        st.write(f"**{i}. {display_name}**")
                        st.progress(float(probability))
                        st.write(f"Säkerhet: {probability*100:.2f}%")
                        st.write("---")
                else:
                    st.warning(f"Inga prediktioner över {confidence_threshold*100:.0f}% säkerhet.")

                # Visa top prediktion med stor text
                top_class = predictions[0][1].replace('_', ' ').title()
                top_prob = predictions[0][2]
                st.success(f"**Bästa gissning:** {top_class}")
                st.metric("Säkerhet", f"{top_prob*100:.2f}%")
                
                # Spara till historik
                if st.button(f"Spara till historik", key=f"save_{idx}"):
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': uploaded_file.name,
                        'model': selected_model,
                        'top_prediction': top_class,
                        'confidence': f"{top_prob*100:.2f}%",
                        'all_predictions': filtered_preds
                    })
                    st.success("Sparat till historik!")
            
            st.markdown("---")
    else:
        # Visa instruktioner
        st.info("👆 Ladda upp en eller flera bilder för att börja klassificera!")

with tab2:
    st.header("Använd kamera")
    st.write("Ta ett foto direkt med din kamera!")
    
    camera_photo = st.camera_input("Ta en bild")
    
    if camera_photo:
        img = Image.open(camera_photo)
        
        # Applicera filter om aktiverade
        if apply_filters:
            img = apply_image_filters(img, brightness, contrast, sharpness)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tagen bild")
            st.image(img, use_container_width=True)
        
        with col2:
            st.subheader("AI Prediktioner")
            
            with st.spinner('Analyserar bild...'):
                predictions = predict_image(img, model, selected_model, top_k)
            
            st.write("### Resultat:")
            
            filtered_preds = [(name, prob) for _, name, prob in predictions if prob >= confidence_threshold]
            
            if filtered_preds:
                for i, (class_name, probability) in enumerate(filtered_preds, 1):
                    display_name = class_name.replace('_', ' ').title()
                    st.write(f"**{i}. {display_name}**")
                    st.progress(float(probability))
                    st.write(f"Säkerhet: {probability*100:.2f}%")
                    st.write("---")
            
            top_class = predictions[0][1].replace('_', ' ').title()
            top_prob = predictions[0][2]
            st.success(f"**Bästa gissning:** {top_class}")
            st.metric("Säkerhet", f"{top_prob*100:.2f}%")
            
            # Spara till historik
            if st.button("Spara till historik", key="save_camera"):
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'filename': 'Camera Photo',
                    'model': selected_model,
                    'top_prediction': top_class,
                    'confidence': f"{top_prob*100:.2f}%",
                    'all_predictions': filtered_preds
                })
                st.success("Sparat till historik!")

with tab3:
    st.header("Prediktionshistorik")
    
    if st.session_state.prediction_history:
        st.write(f"**Totalt antal prediktioner:** {len(st.session_state.prediction_history)}")
        
        # Konvertera till DataFrame
        df = pd.DataFrame([
            {
                'Tidpunkt': item['timestamp'],
                'Filnamn': item['filename'],
                'Modell': item['model'],
                'Prediktion': item['top_prediction'],
                'Säkerhet': item['confidence']
            }
            for item in st.session_state.prediction_history
        ])
        
        # Visa tabell
        st.dataframe(df, use_container_width=True)
        
        # Statistik
        col1, col2, col3 = st.columns(3)
        
        with col1:
            most_common = df['Prediktion'].mode()
            if len(most_common) > 0:
                st.metric("Vanligaste klass", most_common[0])
        
        with col2:
            models_used = df['Modell'].nunique()
            st.metric("Modeller använda", models_used)
        
        with col3:
            total_predictions = len(df)
            st.metric("Totala prediktioner", total_predictions)
        
        # Rensa historik knapp
        if st.button("Rensa historik"):
            st.session_state.prediction_history = []
            st.rerun()
        
        # Ladda ner historik som CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Ladda ner historik (CSV)",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Ingen historik ännu. Börja klassificera bilder och spara dem till historiken!")

# Footer
st.sidebar.markdown("---")
st.sidebar.header("ℹInformation")

with st.sidebar.expander("Om modellerna"):
    st.write(f"""
    **Aktuell modell:** {selected_model}
    
    **Tillgängliga modeller:**
    """)
    for model_name, info in MODEL_INFO.items():
        st.write(f"- **{model_name}:** {info['description']}")
    
    st.write("""
    ---
    **Gemensamt för alla modeller:**
    - Dataset: ImageNet
    - Antal klasser: 1000
    - Tränade på miljontals bilder
    """)

with st.sidebar.expander("Hur det fungerar"):
    st.write("""
    1. **Ladda upp/ta bild:** Välj eller ta en bild
    2. **Preprocessing:** Bilden ändras till rätt storlek och format
    3. **Prediktion:** CNN-modellen analyserar bilden
    4. **Resultat:** De mest sannolika klasserna visas
    
    **CNN (Convolutional Neural Network)** är specialiserad på bildanalys
    genom att lära sig hierarkiska features från enkla kanter till
    komplexa objekt.
    """)

with st.sidebar.expander("Snabbtangenter"):
    st.write("""
    - `Ctrl/Cmd + Enter`: Kör cell
    - `R`: Uppdatera app
    - `C`: Rensa cache
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Skapad med Streamlit & TensorFlow/Keras")
