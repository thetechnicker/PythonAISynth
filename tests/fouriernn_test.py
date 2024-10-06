# import os
# import unittest
# from context import src
# from src import fourier_neural_network

# import numpy as np
# import tensorflow as tf

# # TODO could use refactor

# path = './tests/tmp/'
# if not os.path.exists(path=path):
#     os.mkdir(path=path)


# class TestStringMethods(unittest.TestCase):
#     # changed the way fourierNN saves the data
#     # TODO: update functon
#     # def test_fouriernn_init(self):
#     #     fourier_nn = fourier_neural_network.FourierNN(
#     #         np.array([(0, 1), (2, 3), (4, 5)]))
#     #     np.testing.assert_array_equal(
#     #         fourier_nn.data, np.array([(0, 1), (2, 3), (4, 5)]))
#     #     self.assertIsNone(fourier_nn.models)
#     #     self.assertIsNotNone(fourier_nn.current_model)

#     def test_relu_aprox(self):
#         data = [(x, tf.keras.activations.relu(x))
#                 for x in np.linspace(np.pi, -np.pi, 100)]
#         fourier_nn = fourier_neural_network.FourierNN(data)
#         # self.assertEqual(fourier_nn.data, data)
#         fourier_nn.train(quiet=True)
#         test_data = np.linspace(np.pi, np.pi, fourier_nn.SAMPLES)
#         out = fourier_nn.predict(test_data)
#         wanted_data=np.array([tf.keras.activations.relu(x) for x in test_data]).reshape((fourier_nn.SAMPLES, 1))
#         #print(wanted_data)
#         np.testing.assert_almost_equal(out, wanted_data, decimal=1)

#     def test_sine_of_relu_aprox(self):
#         data = [(x, np.sin(x * tf.keras.activations.relu(x)))
#                 for x in np.linspace(np.pi, -np.pi, 100)]
#         fourier_nn = fourier_neural_network.FourierNN(data)
#         # self.assertEqual(fourier_nn.data, data)
#         fourier_nn.train(quiet=True)
#         test_data = np.linspace(np.pi, np.pi, fourier_nn.SAMPLES)
#         #print(data)
#         out = fourier_nn.predict(test_data)
#         wanted_data=np.array([np.sin(x*tf.keras.activations.relu(x)) for x in test_data]).reshape((fourier_nn.SAMPLES, 1))
#         #print(wanted_data)
#         #np.testing.assert_almost_equal(data, wanted_data)
#         np.testing.assert_almost_equal(out, wanted_data, decimal=1)


# if __name__ == '__main__':
#     unittest.main()
# # # Test the class with custom data points
# # data = [(x, np.sin(x * tf.keras.activations.relu(x)))
# #         for x in np.linspace(np.pi, -np.pi, 100)]
# # fourier_nn = FourierNN(data)
# # fourier_nn.train()
