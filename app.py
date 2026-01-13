import streamlit as st                            # Importujemy framework do budowy interfejsu.
import pandas as pd                               # Do obsługi danych wejściowych dla modelu.
from pycaret.regression import load_model, predict_model # Do wczytania i użycia modelu.
import os                                         # Do pobierania kluczy z .env.
from openai import OpenAI                         # Do komunikacji z LLM.
from langfuse.openai import openai                # Wrapper Langfuse dla monitoringu OpenAI.
from dotenv import load_dotenv                    # Do wczytania poświadczeń.
import json                                       # Importujemy bibliotekę do obsługi formatu JSON.
import time                                       # Importujemy do formatowania czasu.

load_dotenv()                                     # Ładujemy klucze.

# 1. Konfiguracja modelu i strony
model = load_model('marathon_model_v1')           # Wczytujemy zapisany wcześniej lokalnie model.

st.title("🏃 Przewodnik Półmaratończyka")          # Ustawiamy tytuł aplikacji.
st.write("Powiedz mi coś o swoim bieganiu, a przewidzę Twój czas!")

# 2. Interfejs użytkownika
user_input = st.text_area("Przedstaw się (np. 'Jestem facetem, mam 30 lat i biegam 5km w 22:30')")

if st.button("Szacuj czas"):                      # KLUCZOWY BLOK: Wszystko poniżej musi być wcięte!
    if user_input:
        # 3. LLM Parsing - Wersja z wymuszonym logowaniem do Langfuse
        client = OpenAI()
        
        prompt = f"""Wyodrębnij dane z tekstu biegacza. 
        Zwróć dane w formacie JSON o kluczach: "plec" (M/K), "wiek" (liczba), "czas_5km" (MM:SS).
        Tekst użytkownika: {user_input}"""

        messages_list = [
            {"role": "system", "content": "Jesteś precyzyjnym parserem danych. Zwracasz wyłącznie czysty JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Wywołanie modelu
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages_list,
                response_format={ "type": "json_object" }, 
                temperature=0,
                name="marathon-data-extraction"
            )
            
            raw_data = response.choices[0].message.content

            # 4. Przetwarzanie i Walidacja
            if not raw_data:
                st.error("AI nie zwróciło danych.")
            else:
                data = json.loads(raw_data)
                
                # SEKCJA DEBUG
                st.info(f"DEBUG: AI wyodrębniło: {data}")

                plec_raw = str(data.get('plec', '')).upper()
                wiek_raw = data.get('wiek')
                czas_raw = data.get('czas_5km')
                
                errors = []

                # Walidacja płci
                if plec_raw not in ['M', 'K']:
                    errors.append(f"Nie rozpoznano płci ('{plec_raw}').")

                # Walidacja wieku
                try:
                    wiek_int = int(wiek_raw)
                    if wiek_int < 15 or wiek_int > 100:
                        errors.append(f"Wiek {wiek_int} poza skalą.")
                except:
                    errors.append("Wiek musi być liczbą.")

                # Funkcja czasu
                def local_time_to_seconds(t):
                    try:
                        parts = list(map(int, str(t).split(':')))
                        if len(parts) == 2: return parts[0] * 60 + parts[1]
                        if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
                        return None
                    except: return None

                sekundy_5km = local_time_to_seconds(czas_raw)
                
                # --- WALIDACJA ZAKRESU CZASU (12 min - 60 min) ---
                if sekundy_5km is None:
                    errors.append(f"Niepoprawny format czasu ({czas_raw}). Użyj MM:SS.")
                elif sekundy_5km < 720: 
                    errors.append(f"Czas 5km ({czas_raw}) jest nierealny (poniżej 12:00).")
                elif sekundy_5km > 3600: 
                    errors.append(f"Czas 5km ({czas_raw}) jest zbyt długi (powyżej 60:00). To aplikacja dla biegaczy!")

                if errors:
                    for err in errors:
                        st.error(f"❌ {err}")
                else:
                    # --- 5. Przygotowanie danych dla modelu ---
                    input_df = pd.DataFrame([{
                        'plec': 1 if plec_raw == 'M' else 0,
                        'wiek': wiek_int,
                        '5km_s': sekundy_5km
                    }])

                    # --- 6. Predykcja i wyświetlenie wyniku ---
                    with st.spinner('Trwa obliczanie wyniku...'):
                        prediction = predict_model(model, data=input_df)
                        result_seconds = prediction['prediction_label'].iloc[0]
                        
                        formatted_time = time.strftime('%H:%M:%S', time.gmtime(result_seconds))
                        
                        st.divider()
                        st.success(f"### Twój szacowany czas to: {formatted_time}")
                        st.balloons()

        except Exception as e:
            st.error(f"Błąd krytyczny: {e}")