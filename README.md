# Proyecto de Servidor de Detección y Almacenamiento de Rostros con Flask y MongoDB

¡Bienvenido al repositorio de nuestro emocionante proyecto de servidor de detección y almacenamiento de rostros! En este proyecto, hemos creado una aplicación utilizando Python con el framework Flask, que tiene la capacidad de detectar rostros en imágenes, recortarlos y cifrarlos para garantizar la seguridad de los datos sensibles. Además, hemos integrado un sistema de almacenamiento en una base de datos MongoDB para conservar la información de los rostros detectados.

## Características destacadas

- **Detección de Rostros:** Utilizamos técnicas avanzadas de procesamiento de imágenes y la librería de detección de rostros para identificar y delinear rostros en las imágenes subidas a la aplicación.

- **Recorte y Codificación:** Una vez que se detecta un rostro en una imagen, el servidor recorta cuidadosamente la región del rostro y la codifica, lo que garantiza que la información personal se mantenga segura y privada.

- **Almacenamiento en MongoDB:** Implementamos un sistema de base de datos utilizando MongoDB para almacenar de manera eficiente la información de los rostros detectados. Esto permite un acceso rápido y confiable a los datos cuando sea necesario.

- **Interfaz de Usuario Amigable:** Diseñamos una interfaz web sencilla y fácil de usar que permite a los usuarios cargar imágenes, ver los rostros detectados y gestionar su almacenamiento en la base de datos.

## Instrucciones de Uso

1. **Clonar el Repositorio:** Comienza por clonar este repositorio en tu máquina local utilizando el comando:

   ```
   git https://github.com/PePeWee07/IA-Facial
   ```

2. **Instalar Dependencias:** Accede al directorio del proyecto e instala las dependencias necesarias ejecutando:

   ```
   pip install -r requirements.txt
   ```

3. **Configurar la Base de Datos MongoDB:** Asegúrate de tener MongoDB instalado y en funcionamiento. Actualiza la configuración de conexión en el archivo `config.py` con los detalles de tu base de datos.

4. **Ejecutar la Aplicación:** Ejecuta la aplicación utilizando el siguiente comando:

   ```
   python service.py
   ```

5. **Acceder a la Interfaz de Usuario:** Abre tu navegador web y navega a `http://localhost:5000` para acceder a la interfaz de usuario. Desde allí, podrás cargar imágenes y ver los resultados de la detección de rostros.

## Contribución

¡Agradecemos contribuciones de la comunidad! Si deseas mejorar este proyecto, añadir nuevas características o corregir errores, no dudes en hacer un fork del repositorio y enviar un pull request.

## Licencia

Este proyecto se encuentra bajo la licencia MIT. Puedes encontrar más detalles en el archivo `LICENSE`.

Esperamos que este proyecto te resulte útil y educativo. Si tienes alguna pregunta o sugerencia, no dudes en abrir un issue en el repositorio.

¡Diviértete explorando y desarrollando con nuestro proyecto!

**El Equipo de Desarrollo**

Jose Roman, 
Gustavo Yansa, 
Jorge tenezaca
