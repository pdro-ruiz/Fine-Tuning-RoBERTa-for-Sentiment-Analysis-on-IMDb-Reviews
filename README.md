# **IMDb Sentiment Analysis with RoBERTa**  

## 📖 **Descripción**  

Este proyecto tiene como objetivo demostrar las capacidades de **fine-tuning** del modelo **RoBERTa (Robustly optimized BERT approach)** en el análisis de sentimientos de reseñas de películas de **IMDb**.  

El objetivo es **clasificar las reseñas como positivas o negativas**, aprovechando la potencia de **Transfer Learning** en modelos de lenguaje preentrenados.  

<p align="center">
  <img src="Fine-Tuning RoBERTa for Sentiment Analysis on IMDb Reviews.jpg" alt="Sentiment Analysis with RoBERTa">
</p>  

## 🎯 **¿Por qué RoBERTa?**  

Escogimos **RoBERTa** porque ofrece mejoras significativas sobre el modelo original de BERT:  
✅ Preentrenado en un conjunto de datos mucho más grande.  
✅ Optimizaciones en el proceso de entrenamiento para mejorar el rendimiento.  
✅ Accesibilidad gratuita sin necesidad de API Keys.  

---

## 🚀 **Flujo de Trabajo**  

### 1️⃣ **Carga y Preprocesamiento de Datos**  
📌 **IMDb Dataset** cargado desde `datasets` de Hugging Face.  
📌 **Tokenización** con `RoBERTaTokenizer`, truncando y rellenando (`padding`) para entradas uniformes.  

### 2️⃣ **Fine-Tuning del Modelo**  
📌 **RoBERTaForSequenceClassification** ajustado sobre el conjunto de datos de IMDb.  
📌 Optimización de hiperparámetros (`batch_size`, `learning_rate`, `epochs`).  
📌 Ajuste de pesos para mejorar rendimiento en análisis de sentimientos.  

### 3️⃣ **Evaluación del Modelo**  
📌 Cálculo de métricas clave:  
   - **F1-score**  
   - **Precisión (`Precision`)**  
   - **Recall**  
   - **Loss de validación**  

---

## 📂 **Estructura del Proyecto**  

```
├── preprocess.py   # Preprocesamiento del dataset IMDb
├── train.py        # Entrenamiento del modelo RoBERTa
├── evaluate.py     # Evaluación del modelo
├── results/        # Carpeta donde se guardan los modelos entrenados
└── README.md       # Documentación del proyecto
```

---

## 🔥 **Retos y Adaptaciones**  

🔹 **Limitaciones de Recursos:**  
  - **Fine-tuning** de modelos grandes como RoBERTa requiere alta capacidad de cómputo.  
  - Se optimizaron los **batch sizes** y la carga de datos para evitar saturación.  

🔹 **Manejo del Desbalanceo de Clases:**  
  - Aunque IMDb está relativamente balanceado, fue necesario ajustar el entrenamiento para evitar **overfitting**.  

🔹 **Longitud de los Textos:**  
  - Las reseñas largas requerían **tokenización eficiente** para evitar la pérdida de información importante.  

---

## 📊 **Resultados**  

Obtuvimos un rendimiento sobresaliente en el análisis de sentimientos:  

| **Métrica**           | **Valor**    |
|-----------------------|-------------|
| **Loss de Validación** | 0.2524      |
| **Precisión (`Accuracy`)** | 95.39% |
| **F1-score**         | 0.9541      |
| **Precisión (`Precision`)** | 0.9497      |
| **Recall**           | 0.9586      |

Estos resultados indican que **RoBERTa ofrece un balance óptimo entre precisión y generalización**, siendo altamente confiable para el análisis de sentimientos en reseñas de películas.  

---

## 🏁 **Conclusión**  

Este proyecto demuestra el **poder del aprendizaje por transferencia (Transfer Learning)** y el **fine-tuning** para la clasificación de sentimientos.  

✅ **Utilizando RoBERTa**, logramos **alta precisión y métricas equilibradas** en la clasificación de reseñas de IMDb.  
✅ **Optimización de entrenamiento y procesamiento** permitió lograr un modelo eficiente con buen rendimiento.  

💡 **Posibles mejoras:**  
- Implementación de **GridSearchCV** para afinar hiperparámetros.  
- Uso de **DistilRoBERTa** para reducir el tamaño del modelo sin perder rendimiento.  
- Aplicación de **Data Augmentation** en reseñas de baja representatividad.  

---

## 🛠 **Requisitos y Configuración**  

Asegúrate de instalar los siguientes paquetes antes de ejecutar el código:  
```bash
pip install transformers datasets torch sklearn numpy
```

---

## 💬 **Agradecimientos**  

Este proyecto fue desarrollado utilizando:  
🔹 **Hugging Face Transformers**  
🔹 **Datasets**  
🔹 **PyTorch**  

📢 **Si te resultó útil, dale un ⭐ al repositorio y sígueme en GitHub!** 🚀  

---

## 🔗 **Contacto**  

📌 **[Web](https://pdroruiz.com/)**  
📌 **[GitHub](https://github.com/pdro-ruiz)**  
📌 **[Kaggle](https://www.kaggle.com/pdroruiz)**  
📌 **[LinkedIn](https://www.linkedin.com/in/)**  

¡Gracias por leer! **Nos vemos en el siguiente proyecto!** 👋  

---

Espero que este README sea útil y atractivo para compartir en plataformas como GitHub o Kaggle. 🚀
