# vtbhack2020-ml

<div display="inline">
<img src="https://github.com/IBISolutions/vtbhack2020-ml/blob/main/img/Мокап1.png" width="150px">
<img src="https://github.com/IBISolutions/vtbhack2020-ml/blob/main/img/Мокап2.png" width="150px">
<img src="https://github.com/IBISolutions/vtbhack2020-ml/blob/main/img/Мокап3.png" width="150px">
<img src="https://github.com/IBISolutions/vtbhack2020-ml/blob/main/img/Мокап4.png" width="150px">
<img src="https://github.com/IBISolutions/vtbhack2020-ml/blob/main/img/Мокап5.png" width="150px">
</div>

## Предисловие
У мобильных разработчиков стоит задача, разработать мобильное приложение, которое через камеру смартфона определяет марку автомобиля. Они используют предоставленные банковские API классификации марок автомобилей с ограничением на 14 определенных марок:
- Mazda 6 sedan
- Mazda 3 sedan
- Cadillac ESCALADE
- Jaguar F-PACE
- BMW 5 sedan
- KIA Sportage
- Chevrolet Tahoe
- KIA K5 sedan
- Hyundai Genesis sedan
- Toyota Camry
- Mercedes A sedan
- Land Rover RANGE ROVER VELAR
- BMW 3 sedan
- KIA Optima sedan

## Задача
На основе предоставленного дата-сета из 5 разных марок автомобилей, обучить модель классификации автомобилей и предоставить результат back-end разработчику, которые реализует сервис распознавания и к нему API по <a href="https://drive.google.com/drive/folders/1DVFjeWS7D6l7P63hlT4ljRsoE8lRh-Nq" target="_blank">спецификации</a>, для дальнейшего использования API мобильного разработчика.

## Дата-сеты
Дата-сет содержит следующие марки автомобилей, по ~1000 фотографий .jpeg в каждой:
- <a href="https://drive.google.com/drive/folders/1qtqKl7UBOVWQs0AP0o4YT4GZmf9UrOnS" target="_blank">Hyundai SOLARIS</a>
- <a href="https://drive.google.com/drive/folders/1sUoJIGykcR5savunmTXirVoWN-Oh6e_L" target="_blank">KIA Rio</a>
- <a href="https://drive.google.com/drive/u/1/folders/1pYoMRonIo6oPKt0ntD3QJtooHtKh_9Su" target="_blank">Volkswagen Polo</a>
- <a href="https://drive.google.com/drive/u/1/folders/1nJy9TXFAPNzs1Gg2jNzAbbamsfURqtle" target="_blank">Volkswagen Tiguan</a>
- <a href="https://drive.google.com/drive/folders/1MF8rlcXMSldHHD0Ec8Yvvvjvf37wJ691" target="_blank">Skoda Octavia</a>

## Описание модели

**Ссылка на модель .h5:** <br>
**Новая актуальная версия:** https://yadi.sk/d/FFkWaihY-5RTrQ <br>
Прошлая версия: https://yadi.sk/d/qPnhh3oPkIL67A <br>
Необходимо поместить в корне проекта в папку **models**.

Для реализации определения машин мы использовали предобученную модель **VGG16** на датасете **ImageNet**. 
Моделирование при помощи фреймворка **Keras** с бэкэндом **TensorFlow**.
Переобучение модели достигалось за счет техники «тонкой настройки»: сохранение весов сети и перенастройка последних полносвязных слоев для классификации наших видов машин. Датасет состоит из 5 классов, мы разделили этот набор на тренировочную и валидационную выборки (75%/25%). Также попробовали обучить модель без валидационной выборки. 

Accuracy ~ 85%. 

## API 
Используются мощности **Яндекс Облака** со следующими характеристиками: 
- Платфоррма: Intel Cascade Lake 
- Гарантированная доля vCPU: 100%
- VCPU: 2
- RAM: 4Гб
- Объем HDD: 20Гб
- ОС: Ubuntu 20 LTS

Доступ к API:
http://130.193.51.57:5000/api/car-recognize <br>
В теле запроса должен быть передан {"content" : "IMG BASE64"}. 

# Команда 

Наша команда: 
- Иван Федоров (Product Owner )
- Виктор Волков (IOS Developer)
- Сергей Кудрявцев (Machine Learning)
- Наталья Федорова (Design)
