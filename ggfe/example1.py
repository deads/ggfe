import string_features
from ggfe.core import variables, functions, Variable, Grammar, Environment

grammar = Grammar("helloworld")
X, Y = variables(["X", "Y"])
TransformString = grammar.productions(["TransformString"])
lower, upper = functions(["lower", "upper"], module=string_features)

TransformString[X] = (lower(X) | upper(X))
