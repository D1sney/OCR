import pandas as pd

# Устанавливаем опцию для отображения всех столбцов
pd.set_option('display.max_columns', None)

# Читаем файл без заголовков (или задайте header=0, если есть)
df = pd.read_excel("/home/userus/PycharmProjects/OCR/temp/7567660873_3_6_ЦО_Климовск,_Октябрьская_пл_,_д_4_1.xlsx", header=None)

# Выводим первые 50 строк с отображением всех столбцов
print(df.head(50))