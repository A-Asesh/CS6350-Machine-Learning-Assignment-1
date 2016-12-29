import math

def read_data(filename):
	total_data=[]
	features=[]
	class_label_column=[]
	with open(filename) as lines:
		for line in lines:
			line=line.replace(",","")
			line=line.split()
			total_data.extend(line)
			features.append(line[0][:-1])
			class_label_column.append(line[0][-1])

	return total_data,features,class_label_column 


def find_unique(label_column):
	unique={}
	for value in label_column:
		if value in unique:
			unique[value]+=1
		else:
			unique[value]=1

	return unique

def find_total_entropy(unique_value_of_labels):
	total=0
	total_entropy=0
	unique=find_unique(unique_value_of_labels)
	#print unique_value_of_labels
	for elem in unique:
		total=total+unique[elem]
	for occurence in unique.values():
		total_entropy=total_entropy+(float(occurence)/total)*math.log(float(occurence)/total, 2)*-1

	return total_entropy

def find_entropy(length_of_attr,unique):
	data_entropy=0.0
	for frequency in unique.values():
		data_entropy -= (float(frequency)/length_of_attr) * math.log(float(frequency)/length_of_attr, 2)

	return data_entropy 


def calculate_gain(data,column,entropy,column_number,final_gains):
	unique=find_unique(column)
	occurences={}
	label_for_column=[]
	for i in range(len(column)):
		label_for_column.append(data[i][-1])
	#print label_for_column[1:5]
	#print len(column)," this is columns"
	for i in range(len(column)):
		if column[i] in unique:
			if column[i] in occurences:
				if data[i][-1] in occurences[column[i]]:
					occurences[column[i]][data[i][-1]]+=1
					occurences[column[i]]["total"]+=1
				else:
					occurences[column[i]][data[i][-1]]=1
					occurences[column[i]]["total"]+=1
			else:
				occurences[column[i]]={data[i][-1]:1}
				occurences[column[i]]["total"]=1
	info_D=0.0

	for item in occurences:
		occurence_of_item=occurences[item]["total"]
		total=len(column)
		del occurences[item]["total"]
		info_D=info_D+(float(occurence_of_item)/total)*find_entropy(occurence_of_item,occurences[item])
	information_gain=entropy-info_D
	final_gains.append([information_gain,column_number])
	return final_gains

def find_best_split(data,features,labels,columns):
	gain_result=[]
	column_number=0
	entropy=find_total_entropy(labels)
	final_gains=[]
	for i in columns[:-1]:
		column_number+=1
		final_gains=calculate_gain(data,i,entropy,column_number,final_gains)

	best_gain=max(final_gains)
	return best_gain[1]

def fetch_rows(data,best_split_attribute,best_split_number):
	unique=find_unique(best_split_attribute)
	associated_tuples=[]
	for val in unique:
		for elem in range(len(data)):
			if val==data[elem][best_split_number]:
				associated_tuples.append([val,data[elem][-1],elem])

	values_dict={}
	for elem in associated_tuples:
		if (elem[0]) in values_dict:
			values_dict[elem[0]].append(elem[2])
		else:
			values_dict[elem[0]]=[elem[2]]

	associated_tuples_list=[]
	for a in values_dict:
		associated_tuples_list.append([a,values_dict[a]])

	return associated_tuples_list

def getData(key,data,best_split_number):
	new_derived_data=[]
	derived_data=[]
	for row in key[1]:
		derived_data.append(data[row])
	#print derived_data[1:5]
	for i in derived_data:
	#	print type(i)
		new_derived_data.append(i[:best_split_number]+i[best_split_number+1:])
	#print len(new_derived_data)
	return new_derived_data


def make_decision_tree(total_data,features,class_label_column,column):
	unique_value_of_labels=find_unique(class_label_column)
	max=0
	default_value=None
	for keys in unique_value_of_labels:
		if unique_value_of_labels[keys]>max:
			max=unique_value_of_labels[keys]
			default_value=keys

	if not total_data or (len(features) - 1) <= 0:
		return	default_value

	elif class_label_column.count(class_label_column[0])==len(class_label_column):
		return class_label_column[0]

	else:
		best_split_number=find_best_split(total_data,features,class_label_column,column)-1		
		best_split_attribute= column[best_split_number]
		decision_tree={best_split_number:{}}
		associated_tuples_list=fetch_rows(total_data,best_split_attribute,best_split_number)
		
		for key in associated_tuples_list:
			derived_list_of_associated_values=[]
			new_derived_data=getData(key,total_data,best_split_number)
			new_derived_lables=[i[-1] for i in new_derived_data]
			for i in new_derived_data:
				derived_list_of_associated_values.append(i[:-1])
			derived_column=[[0 for x in range(len(new_derived_data))] for x in range(len(new_derived_data[0]))]
			for i in range(len(new_derived_data)):
				for j in range(len(new_derived_data[0])):
					derived_column[j][i]=new_derived_data[i][j]

			#print len(new_derived_data),len(derived_list_of_associated_values), len(derived_column)
			
			decision_tree[best_split_number][key[0]]=make_decision_tree(new_derived_data,derived_list_of_associated_values,new_derived_lables,derived_column)

	return decision_tree

def classify(record,tree):
	if type(tree) == type("string"):
		return tree
	else:
		key = tree.keys()[0]
		if record[key] in tree[key]:
			t = tree[key][record[key]]
			new_record=record[:key]+record[key+1:]
		else:
			return "not found"
		return classify(new_record, t)


def tester(test_data,tree):
	tested=[]
	for line in test_data:
		test=classify(line,tree)
		tested.append(test)
	return tested

def find_accuracy(classified,test_features_test_label_columns):
	total=0
	hits=0
	accuracy=0
	while total<len(classified):
		
		if classified[total]==test_features_test_label_columns[total]:
			hits+=1
		total+=1

	accuracy=float(hits)/total
	return accuracy

def main():
	classified=[]
	filename="trainingB.data"
	total_data,features,class_label_column=read_data(filename)
	column=[[0 for x in range(len(total_data))] for x in range(len(total_data[0]))]
	for i in range(len(total_data)):
		for j in range(len(total_data[0])):
			column[j][i]=0
	for i in range(len(total_data)):
		for j in range(len(total_data[0])):
			column[j][i]=total_data[i][j]
	
	tree=make_decision_tree(total_data,features,class_label_column,column)
	#print tree
	filename="trainingA.data"
	test_data,test_features,test_label_columns=read_data(filename)
	classified=tester(test_data,tree)
	accuracy=find_accuracy(classified,test_label_columns)
	print accuracy, "this is accuracy"




main()


