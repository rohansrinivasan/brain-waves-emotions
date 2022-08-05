def preprocess_inputs(df):
  df = df.copy()
  df['label'] = df['label'].replace(label_mapping)

  y = df['label'].copy()
  x = df.drop('label', axis=1).copy()

  x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=123)

  return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = preprocess_inputs(data)

x_train