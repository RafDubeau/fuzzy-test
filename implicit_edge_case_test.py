import implicit
import numpy as np
from scipy.sparse import csr_matrix


user_items_dense = np.zeros((3, 4))
user_items_sparse = csr_matrix(user_items_dense)


model = implicit.als.AlternatingLeastSquares(factors=10, iterations=20)
model.fit(user_items_sparse)

print(model.user_factors)

print("----")

print(model.item_factors)
