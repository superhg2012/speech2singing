import torch
import torch.nn as nn
class GeneralModel(nn.Module):
    """
    General model feed by yaml config
    """
    def __init__(self, module_list, arg_list):
        """
        PARAMS:
        module_list: module_input from utils/process_yaml_model.py
        arg_list: arg_input from utils/process_yaml_model.py
        """

        super(GeneralModel, self).__init__()
        #self.module_list = module_list
        self.arg_list = arg_list
        self.module_list = nn.ModuleList(module_list)
    def forward(self, *arg):
        arg = list(arg)
        for i in range(len(self.module_list)):
            module = self.module_list[i]
            arg_input = self.arg_list[i]
            #print ("general_modle: ", module)
            input = tuple(arg[index] for index in arg_input)
            #print ("now: ", (input[0]).size())
            output = (module(*input),)
            for i in range(len(output)):
                arg[arg_input[i]] = output[i]
        
        if len(arg) == 1:
            
            return output[0]    
            #print ("----------------------")
        else:
            return arg
    def inference(self, *arg):
        arg = list(arg)
        for i in range(len(self.module_list)):
            module = self.module_list[i]
            arg_input = self.arg_list[i]
            #print ("general_modle: ", module)
            input = tuple(arg[index] for index in arg_input)
            #print ("now: ", (input[0]).size())
            output = (module.inference(*input),)
            output = list(*output)
            #print ("----------------------")
            for i in range(len(output)):
                arg[arg_input[i]] = output[i]
        if len(arg) == 1:
            return arg[0]
        else:
            return arg