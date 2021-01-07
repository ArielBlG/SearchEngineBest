# import pickle
#
# file = open("inverted_index.pkl",'rb')
# indexNoDash = pickle.load(file)
# postingNoDash = pickle.load(file)
# file.close()
#
# file = open("inverted_indexWithDash.pkl",'rb')
# indexWithDash = pickle.load(file)
# postingWithDash = pickle.load(file)
# file.close()
#
# indexNoDash = list(indexNoDash.items())
# indexNoDash = sorted(indexNoDash, key=lambda item: item[1], reverse=True)
# indexWithDash = list(indexWithDash.items())
# indexWithDash = sorted(indexWithDash, key=lambda item: item[1], reverse=True)
# for term in indexNoDash:
#     if term not in indexWithDash:
#         print(term)
