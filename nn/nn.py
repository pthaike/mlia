# !/bin/python

import random
from numpy import *
import math
import pdb


class NeuronNetwork:
	def __init__(self, num_inputs, num_hidden, num_outputs, rate = 0.5,
		hidden_layer_weights = None, hidden_layer_bias = None,
		output_layer_weights = None, output_layer_bias = None):

		self.num_inputs = num_inputs
		self.rate = rate
		self.hidden_layer = NueronLayer(num_hidden, hidden_layer_bias)
		self.output_layer = NueronLayer(num_outputs, output_layer_bias)

		self.init_input_to_hidden_weight(hidden_layer_weights)
		self.init_hidden_to_output_weight(output_layer_weights)


	def init_input_to_hidden_weight(self, hidden_layer_weights):
		# pdb.set_trace()
		wix = 0
		for i in range(len(self.hidden_layer.neurons)):
			for j in range(self.num_inputs):
				if hidden_layer_weights:
					self.hidden_layer.neurons[i].weight.append(hidden_layer_weights[wix])
				else:
					self.hidden_layer.neurons[i].weight.append(random.random())
				wix += 1


	def init_hidden_to_output_weight(self, output_layer_weights):
		wix = 0
		for i in range(len(self.output_layer.neurons)):
			for j in range(len(self.hidden_layer.neurons)):
				if output_layer_weights:
					self.output_layer.neurons[i].weight.append(output_layer_weights[wix])
				else:
					self.output_layer.neurons[i].weight.append(random.random())
				wix += 1


	def feed_forward(self, inputs):
		hidden_layer_output = self.hidden_layer.feed_forward(inputs)
		return self.output_layer.feed_forward(hidden_layer_output)


	def train(self, training_inputs, training_outputs):
		self.feed_forward(training_inputs)

		# output delta
		output_delta = [0] * len(self.output_layer.neurons)
		for i in range(len(output_delta)):
			output_delta[i] = self.output_layer.neurons[i].cal_delta(training_outputs[i])

		# hidden layer delta: w * delta * f'(z)
		hidden_delta = [0] * len(self.hidden_layer.neurons)
		for i in range(len(hidden_delta)):
			w_p_delta = 0
			for j in range(len(self.output_layer.neurons)):
				w_p_delta += output_delta[j] * self.output_layer.neurons[j].weight[i]
			hidden_delta[i] = w_p_delta * self.hidden_layer.neurons[i].cal_derivative()

		# update weight
		# update output layer weight
		for i in range(len(self.output_layer.neurons)):
			for j in range(len(self.output_layer.neurons[i].weight)):
				self.output_layer.neurons[i].weight[j] -= self.rate * output_delta[i] * self.output_layer.neurons[i].inputs[j]

				# update bias
				self.output_layer.neurons[i].bias -= self.rate * output_delta[i]

		# update hidden layer weight
		for i in range(len(self.hidden_layer.neurons)):
			for j in range(len(self.hidden_layer.neurons[i].weight)):
				self.hidden_layer.neurons[i].weight[j] -= self.rate * hidden_delta[i] * self.hidden_layer.neurons[i].inputs[j]

				# update bias
				self.hidden_layer.neurons[i].bias -= self.rate * hidden_delta[i]


	def total_error(self, train):
		err = 0
		for t in range(len(train)):
			inputs, outputs = train[t]
			self.feed_forward(inputs)
			for i in range(len(outputs)):
				err += self.output_layer.neurons[i].cal_error(outputs[i])
		return err


# layer of neuron network
class NueronLayer:
	def __init__(self, num_neurons, bias):
		self.bias = bias if bias else random.random()
		self.neurons = []
		for i in range(num_neurons):
			self.neurons.append(Neuron(self.bias))

	def feed_forward(self, x):
		self.outputs = []
		for n in self.neurons:
			self.outputs.append(n.cal_output(x))
		return self.outputs


class Neuron:
	def __init__ (self, bias):
		self.bias = bias
		self.weight = []

	def cal_input(self, x):
		y = 0
		# print len(x), len(self.weight)
		for i in range(len(x)):
			y += self.weight[i] * x[i]
		return y + self.bias


	def cal_output(self, x):
		self.inputs = x
		self.output = self.squash(self.cal_input(x))
		return self.output
	
	def squash(self, total_input):
		return 1 / (1 + math.exp(-total_input))

	# Mean Square Error
	def cal_error(self, target):
		return 0.5 * (self.output - target) ** 2

	# for Mean Square Error, the delta of output is E / z = (y - t)
	def cal_delta(self, target):
		return self.output - target


	# for sigmoid function, the derivative of it is f'(x) = f(x) * (1 - f(x))
	def cal_derivative(self):
		return self.output * (1 - self.output)



if __name__ == '__main__':
	nn = NeuronNetwork(2,2,2,hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
	for i in range(100):
		nn.train([0.05, 0.1], [0.01, 0.99])
		print(i, round(nn.total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))