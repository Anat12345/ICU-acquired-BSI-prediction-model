

class Chartevent_param_data(object):


    def __init__(self, itemid_list, param_name):
        self.itemid_list = itemid_list
        self.param_name = param_name
        self.df_list = list()

    def add_df (self,df):
        self.df_list.append (df)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)