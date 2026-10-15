import numpy as np

class Model : 
  def __init__(self,n_feature,out_features,alpha):
    self.w = np.zeros((n_feature, out_features))
    self.b = np.zeros((1, out_features))
    self.alpha = alpha

  def compute_y_hat(self,X: np.ndarray) -> np.ndarray:
    return np.dot(X,self.w) + self.b

  def compute_gradients(self,X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = X.shape[0]
    y_hat = self.compute_y_hat(X)
    dj_dw = (1 / m) * np.dot(X.T,y_hat - y)
    dj_db = (1 / m) * np.sum(y_hat - y, axis=0, keepdims=True)
    return dj_dw, dj_db

  def compute_mse(self,m: int, X: np.ndarray, y : np.ndarray) -> float:
    return (1/(2*m)) * (np.sum((self.compute_y_hat(X)- y) ** 2))

  def train(self,X: np.ndarray, y: np.ndarray, epochs: int): #batch gradient descent
    m = X.shape[0]
    for i in range(epochs):
      dj_dw, dj_db = self.compute_gradients(X,y)
      self.w = self.w - self.alpha * dj_dw
      self.b = self.b - self.alpha * dj_db
      if i % 100 == 0 :
        print(f"Epochs : {i} MSE : {self.compute_mse(m,X,y)}")
    return self.w, self.b

  def predict(self,X: np.ndarray) -> np.ndarray :
    return np.dot(X,self.w) + self.b
