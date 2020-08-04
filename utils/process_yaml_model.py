import sys
#sys.path.append('../model/')

import yaml
import json
import torch
from layers import *
from general_model import GeneralModel

class YamlModelProcesser():

	def construct_model(self, file_name):
		"""
		Construct model from layers.py
		
		Return:
		
		list of model()
		"""

		with open(file_name, 'r') as stream:
			try:
				config = yaml.safe_load(stream)
			except yaml.YAMLError as exc:
				print (exc)
				return 

		for model in config:
			model_name = list(model.keys())[0]
			module_list = []
			arg_list = []
			#print ("---------------------------")
			#print ("model name: ", model_name)
			for module in model[model_name]:
				#print ("module: ", module)
				layer, input_arg = self.parseString(module)
				module_list.append(layer)
				arg_list.append(input_arg)
			#print ("---------------------------")
		gm = GeneralModel(module_list, arg_list)#.cuda()
		return gm
	def parseString(self, moduleDict, recursive = True):
		module_name = list(moduleDict.keys())[0]

		name, input_arg = module_name.split(" ")
		params = moduleDict[module_name]
		
		if (params) == None:

			layer = eval(name)()
			return layer, tuple(map(int, input_arg.split(",")))
		
		elif (recursive == True):
			params = self.block2gm(params)
		layer = eval(name)(**params)

		return layer, tuple(map(int, input_arg.split(",")))
		
	def block2gm(self, params):
		if isinstance(params, dict) == False:
			return params
		for key in params:
			if key[:5] == "block":
				params[key] = self.blocklist2gm(params[key])
		return params
	def blocklist2gm(self, params_in_block):
		"""
		params_in_block: a list of blocks
		
		return:
		
		sub_gm(GeneralModel): general model with these blocks
		"""
		#print ("params_in_block: ", params_in_block)
		#print ("len of params in block:", len(params_in_block))
		#print ("construct_model: ", params_in_block)
		for i in range(len(params_in_block)):
			#print ("block_dict", block_dict)
			block_dict = params_in_block[i]
			for key in block_dict.keys():

				block_dict[key] = self.block2gm(block_dict[key])
			#print ("outog key: ", block_dict)
			layer, input_arg = self.parseString(block_dict, recursive = False)
			block_dict = GeneralModel([layer], [input_arg])
			params_in_block[i] = block_dict
		#print ("end in here:", params_in_block)
		return nn.Sequential(*params_in_block)
			
if __name__=="__main__":
	ymp = YamlModelProcesser()
	gm = ymp.construct_model("../config/test.yaml")
	
	x = torch.rand(10,32,32)
	x = gm(x)
	#dims = x.size()
	#x = x.view(-1, 32)
	#x = x.view(*dims[:-1], 32)
	print (x[0].size())
    #x = self.linear_layer(x)
    