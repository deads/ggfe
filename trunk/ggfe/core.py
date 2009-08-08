# Grammar-Guided Feature Extraction (GGFE)
#
# Author:   Damian Eads
#
# File:     core.py
#
# Purpose:  This module contains functions and data structures for
#           specifying generative grammars and generating features from
#           grammars.
#
# Date:     August 2, 2009

import types
import string
import numpy as np

class Environment:
    """
    Encapsulates information about global variables and local variables on the
    call stack.
    """

    def __init__(self):
        "Initializes a new empty call stack."
        self.globals = {}
        self.stack = []

    def num_frames(self):
        "Returns the number of frames in the call stack."
        return len(self.stack)

    def new_frame(self, dict={}):
        "Creates a new frame on the call stack."
        self.stack.append(dict)

    def delete_frame(self):
        "Deletes the top most frame on the call stack."
        self.stack = self.stack[:-1]

    def frame_lookup(self, name, level=0):
        "Looks up a variable in the current top-most frame, which is level=0"
        retval = None
        if len(self.stack) > 0:
            retval = self.stack[-(level+1)].get(name, None)
        return retval

    def stack_lookup(self, name):
        """
        Looks up a variable in all the frames, starting at the top, returning
        the first match.
        """
        retval = None
        for level in xrange(0, len(stack)):
            retval = self.frame_lookup(name, level)
            if retval is not None:
                break
        return retval

    def complete_lookup(self, name):
        """
        Performs a global variable look-up, then if not successful, performs
        a stack lookup.
        """
        retval = self.globals_lookup(name)
        if retval is None:
            retval = self.stack_lookup(name)
        return retval

    def globals_lookup(self, name):
        """
        Looks up a global variable.
        """
        return self.globals.get(name, None)
            

class RuleList:
    """
    Encapsulates a list of rules of a production.
    """
    def __init__(self):
        """
        Initializes an empty list of rules.
        """
        self.rules = []

    def add_rule(self, rule):
        """
        Adds a rule to this rule list.
        """
        self.rules.append(rule)

    def add_rules(self, rules):
        """
        Adds several rules to this rule list.
        """
        if type(rules) != types.ListType:
            raise TypeError("The list of rules must be a Python list.")
        self.rules.extend(rules)

    def __or__(self, rhs):
        """
        Rules are separated in a list of rules with an or/| symbol.
        """
        self.rules.append(rhs)
        return self

class RuleExpression:
    """
    The base class for all kinds of rule expressions.
    """

    def __init__(self):
        pass

    def __or__(self, rhs):
        """
        Rules are separated in a rule list with an or symbol.
        """
        rule_list = RuleList()
        rule_list.add_rule(self)
        rule_list.add_rule(rhs)
        return rule_list

    def __add__(self, rhs):
        """
        When arithmetic is performed on a rule expression, the arithmetic
        gets caught by this base class and a RuleExpression is created.
        
        This method catches addition.
        """
        if not isinstance(rhs, RuleExpression):
            rhs = LiteralExpression(rhs)
        return BinaryExpression('+', self, rhs)

    def __radd__(self, lhs):
        """
        When arithmetic is performed on a rule expression, the arithmetic
        gets caught by this base class and a RuleExpression is created.
        
        This method catches addition.
        """
        if not isinstance(lhs, RuleExpression):
            lhs = LiteralExpression(lhs)
        return BinaryExpression('+', lhs, self)

    def __sub__(self, rhs):
        """
        When arithmetic is performed on a rule expression, the arithmetic
        gets caught by this base class and a RuleExpression is created.
        
        This method catches subtraction.
        """
        if not isinstance(rhs, RuleExpression):
            rhs = LiteralExpression(rhs)
        return BinaryExpression('-', self, rhs)

    def __rsub__(self, lhs):
        """
        When arithmetic is performed on a rule expression, the arithmetic
        gets caught by this base class and a RuleExpression is created.
        
        This method catches subtraction.
        """
        if not isinstance(lhs, RuleExpression):
            lhs = LiteralExpression(lhs)
        return BinaryExpression('-', lhs, self)

    def __div__(self, rhs):
        """
        When arithmetic is performed on a rule expression, the arithmetic
        gets caught by this base class and a RuleExpression is created.
        
        This method catches division.
        """
        if not isinstance(rhs, RuleExpression):
            lhs = LiteralExpression(rlhs)
        return BinaryExpression('/', self, rhs)

    def __rdiv__(self, lhs):
        """
        When arithmetic is performed on a rule expression, the arithmetic
        gets caught by this base class and a RuleExpression is created.
        
        This method catches division.
        """
        if not isinstance(lhs, RuleExpression):
            lhs = LiteralExpression(lhs)
        return BinaryExpression('/', lhs, self)

    def __mul__(self, rhs):
        """
        When arithmetic is performed on a rule expression, the arithmetic
        gets caught by this base class and a RuleExpression is created.
        
        This method catches multiplication.
        """
        if not isinstance(rhs, RuleExpression):
            rhs = LiteralExpression(rhs)
        return BinaryExpression('*', self, rhs)

    def __rmul__(self, lhs):
        """
        When arithmetic is performed on a rule expression, the arithmetic
        gets caught by this base class and a RuleExpression is created.
        
        This method catches multiplication.
        """
        if not isinstance(lhs, RuleExpression):
            lhs = LiteralExpression(lhs)
        return BinaryExpression('*', lhs, self)

    def __pow__(self, rhs):
        """
        When arithmetic is performed on a rule expression, the arithmetic
        gets caught by this base class and a RuleExpression is created.
        
        This method catches exponentiation.
        """
        if not isinstance(rhs, RuleExpression):
            rhs = LiteralExpression(rhs)
        return BinaryExpression('**', self, rhs)

    def __rpow__(self, lhs):
        """
        When arithmetic is performed on a rule expression, the arithmetic
        gets caught by this base class and a RuleExpression is created.
        
        This method catches exponentiation.
        """
        if not isinstance(lhs, RuleExpression):
            lhs = LiteralExpression(lhs)
        return BinaryExpression('**', lhs, self)

    def __invert__(self):
        """
        Using the tilde ~ (invert) operator on an expression will cause
        the expression to become immediate.
        """
        return Immediate(self)
        
class LambdaCall(RuleExpression):
    """
    An expression object representing the definition of a lambda expression
    as well as its calling. This is useful for reusing computation passed
    to a production. By using lambda expressions, a directed graph representation of programs can be realized.
    
    """

    def __init__(self, arg_names, expr, args):
        self.arg_names = arg_names
        self.expr = expr
        self.args = args            

    def __repr__(self):
        s = "(lambda %s: %s)(%s)" % (
            string.join([str(arg) for arg in self.arg_names], ","),
            self.expr,
            string.join([str(called_arg) for called_arg in self.args], ","))
        return s
                                    
class Variable(RuleExpression):
    """
    Represents local variables and production arguments.
    """

    def __init__(self, name):
        "Creates a new named variable object."
        RuleExpression.__init__(self)
        if type(name) != types.StringType:
            raise ValueError("The symbol's name must be a string.");
        self.name = name

    def get_name(self):
        "Returns the name of this local variable."
        return self.name

    def __repr__(self):
        "Returns a string representation of the variable object, i.e. its name."
        return self.name
    
    def count_local_references(self, cnt_dict):
        "Counts the total number times a variable is referenced in this rule expression before its expanded."
        if self.name in cnt_dict:
            cnt_dict[self.name] += 1
        else:
            cnt_dict[self.name] = 1

    def expand(self, environment):
        "Expands this rule expression. Any productions in this rule expression must expanded as well."
        result = environment.globals_lookup(self.name)
        if result is None:
            result = environment.frame_lookup(self.name)

        if result is None:
            result = self
        return result

    def evaluate(self, environment):
        return environment.complete_lookup(self.name)

class Function(Variable):
    """
    A special kind of variable representing functions.
    """

    def __init__(self, name, module=None):
        "Initializes a new named function."
        Variable.__init__(self, name)
        if type(name) != types.StringType:
            raise ValueError("The function's name must be a string.");
        self.name = name
        self.module = module

    def __call__(self, *args):
        "A function call () is caught during grammar construction."
        cargs = []
        for arg in args:
            if isinstance(arg, RuleExpression):
                cargs.append(arg)
            else:
                cargs.append(LiteralExpression(arg))
        return CallFunction(self, cargs)

class BinaryExpression(RuleExpression):
    """
    Represents a binary expression for binary operators. Every binary
    expression has an operator, a left-hand side expression, and
    a right-hand side expression.
    """

    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def expand(self, environment):
        """
        Returns a new BinaryExpression with the left and right hand
        sides of this binary rule expression expanded.
        """
        lhs_expanded = self.lhs.expand(environment)
        rhs_expanded = self.rhs.expand(environment)
        return BinaryExpression(self.op, lhs_expanded, rhs_expanded)

    def evaluate(self, environment):
        """
        Evaluates this binary expression by evaluating each side then performing
        the binary operation on the result of each side.
        """
        lhs_evaluated = self.lhs.evaluate(environment)
        rhs_evaluated = self.rhs.evaluate(environment)
        result = eval(str(BinaryExpression(self.op, lhs_evaluated, rhs_evaluated)))
        if not isinstance(result, RuleExpression):
            result = LiteralExpression(result)
        return result

    def count_local_references(self, cnt_dict):
        "Counts the total number times a variable is referenced in this rule expression before its expanded."
        self.lhs.count_local_references(cnt_dict)
        self.rhs.count_local_references(cnt_dict)

    def __repr__(self):
        """
        Returns a string representation of this binary expression.
        """
        return "(%s %s %s)" % (self.lhs, self.op, self.rhs)

class LiteralExpression(RuleExpression):

    def __init__(self, val):
        "Initializes a new literal expression with a specific value"
        if isinstance(val, Variable):
            raise ValueError("Cannot initialize a literal expression with a variable")
        self.val = val

    def expand(self, environment):
        "Expands a literal expression, which just returns a copy of itself."
        return LiteralExpression(self.val)

    def evaluate(self, environment):
        "Evaluates a literal expression, which just returns a copy of itself."
        return LiteralExpression(self.val)

    def count_local_references(self, cnt_dict):
        "Counts the total number times a variable is referenced in this rule expression before its expanded. No counts are added because literals do not reference any variables."
        pass

    def __repr__(self):
        """
        Returns a parsable representation of this literal object.
        """
        return str(self.val)

class CallFunction(RuleExpression):

    def __init__(self, fun, args):
        "Returns a new expression object repressenting a function call."
        RuleExpression.__init__(self)
        self.fun = fun
        self.args = args

    def expand(self, environment):
        """
        Expands a function call expression, which returns a new function
        call expression with its arguments expanded.
        """
        expanded_args = []

        # Expand each argument...
        for arg in self.args:
            if isinstance(arg, RuleExpression):
                expanded_args.append(arg.expand(environment))
            else:
                expanded_args.append(LiteralExpression(arg))
        return CallFunction(self.fun, expanded_args)

    def evaluate(self, environment):
        "Evaluates a function call expression."
        evaluated_args = []
        for arg in self.args:
            if isinstance(arg, RuleExpression):
                evaluated_args.append(arg.evaluate(environment))
            else:
                evaluated_args.append(LiteralExpression(arg))
        # Grab the module of the function.
        module = self.fun.module

        # Now grab the function object by name in the modules dictionary.
        fun = self.fun.module.__dict__[self.fun.name]
        
        result = eval(str(CallFunction(Function("fun"), evaluated_args)))
        if not isinstance(result, RuleExpression):
            result = LiteralExpression(result)
        return result

    def count_local_references(self, cnt_dict):
        "Counts the total number times a variable is referenced in this function call expression before its expanded."
        for arg in self.args:
            arg.count_local_references(cnt_dict)

    def __repr__(self):
        "Return a string representation of this function call expression."
        return "%s(%s)" % (self.fun.name, string.join((str(arg) for arg in self.args), ", "))
        
class Expand(RuleExpression):
    """
    A rule expression that represents the expansion of a production rule.
    """

    def __init__(self, prod, args_to_pass):
        RuleExpression.__init__(self)
        self.prod = prod
        if type(args_to_pass) != types.TupleType and type(args_to_pass) != types.ListType:
            self.args_to_pass = [args_to_pass]
        else:
            self.args_to_pass = args_to_pass

    def get_production_name(self):
        """
        Returns the name of the production of expansion.
        """
        return self.prod.name

    def get_arguments_to_pass(self):
        """
        Returns the arguments to pass to the production to expand.
        """
        return self.args_to_pass

    def expand(self, environment):
        """
        Expands a production using any applicable rule for the production,
        selected at random.
        """

        # If only ... are specified as the index arguments, there are 0 production
        # arguments.
        if type(self.args_to_pass) == types.EllipsisType:
            return self.prod(environment=environment)
        expanded_args = []
        for arg in self.args_to_pass:
            if isinstance(arg, RuleExpression):
                expanded_arg = arg.expand(environment)
                if not isinstance(expanded_arg, RuleExpression):
                    expanded_arg = LiteralExpression(expanded_arg)
                expanded_args.append(expanded_arg)
            else:
                expanded_args.append(LiteralExpression(arg))
        return self.prod(*tuple(expanded_args), environment=environment)

    def count_local_references(self, cnt_dict):
        "Counts the total number times a variable is referenced in this rule expression prior to its expansion."
        for arg in self.args_to_pass:
            arg.count_local_references(cnt_dict)

    def __repr__(self):
        """
        A parsable representation of this expansion expression.
        """
        return "%s[%s]" % (self.get_production_name(),
                           string.join([str(arg) for arg in self.get_arguments_to_pass()], ', '))
    
class Production:
    """
    Represents a production in a grammar, which encapsulates one or more rules for
    expanding a non-terminal symbolic expression into a terminal directed graph.
    """

    def __init__(self, name, grammar):
        """
        Creates a new named production of the grammar passed.
        """
        if type(name) != types.StringType:
            raise ValueError("The name of a production must be a string.");
        if not name.isalnum():
            raise ValueError("The name of a production must be alphanumeric.");
        self.name = name
        self.arg_names = None
        self.rules = []
        self.grammar = grammar
        self.grammar.add_production(self)

    def get_name(self):
        """
        Returns the name of this production.
        """
        return self.name

    def __setitem__(self, idx, rules_to_add):
        """
        Rules of a production are specified by indexed assigment.

          Feature[X,Y] = Rule1 | Rule2 | Rule3

        The comma-separated indices are the arguments to the
        production as Variable objects. The rules are expressions which
        may refer to these arguments.
        """
        if type(idx) == types.EllipsisType:
            arg_names = []
        elif type(idx) != types.TupleType and type(idx) != types.ListType:
            arg_names = [idx]
        else:
            arg_names = idx

        if self.arg_names is not None:
            if self.arg_names != arg_names:
                raise ValueError("Inconsistency. Production %s was previously defined with arguments %s but now %s are used." % (self.name, self.arg_names, arg_names))
        else:
            self.arg_names = arg_names
        if isinstance(rules_to_add, RuleList):
            self.rules.extend(rules_to_add.rules)
        elif isinstance(rules_to_add, RuleExpression):
            self.rules.append(rules_to_add)
        else:
            self.rules.append(LiteralExpression(rules_to_add))
#        elif type(rules_to_add) == types.ListType:
#            self.rules.extend(rules_to_add)

    def __getitem__(self, idx):
        """
        When specifying an expansion of a production in rule expressions,
        index the production to be expanded with a comma separated list of
        rule expressions.

           Feature[X] = (Morph[X,RandomSE[...]])
           Morph[X,S] = (erode(X,S)
                         | dilate(X,S))

        """
        if isinstance(idx, types.EllipsisType):
            return Expand(self, ())
        else:
            return Expand(self, idx)

    def __repr__(self):
        """
        Returns a string representation of this production, usually in the
        same syntax as used to parse it.
        """
        return "%s[%s] = %s" % (self.name, string.join([str(arg) for arg in self.arg_names], ","), string.join([str(rule) for rule in self.rules], " | "))


    def __call__(self, *called_args, **kwargs):
        """
        Expands a production, accepting one or more expression objects as
        arguments. These expression objects are referenced with the argument
        names defined for this production.
        """
    
        rule_no = np.random.randint(len(self.rules))
        rule = self.rules[rule_no]
        
        environment = kwargs.get("environment", None)
        if environment is None:
            environment = Environment()

        # If the production is not called with the same number of arguments.
        if (len(called_args) != len(self.arg_names)):
            raise ValueError("Production %s must be expanded with the correct number of arguments, %d, not %d." % (self.name, len(self.arg_names), len(called_args)))

        # Sets the variable reference count to null.
        cnts = {}

        context = {}
        # Count the occurrence of each variable.
        rule.count_local_references(cnts)
        for (argname, called_arg) in zip(self.arg_names, called_args):
            if isinstance(called_arg, Variable):
                val = environment.frame_lookup(called_arg)
                if val is None:
                    context[argname.get_name()] = called_arg
                else:
                    context[argname.get_name()] = val
            else:
                context[argname.get_name()] = called_arg

        print context
        # A lambda expression will be used only if computation needs to be reused.
        lambda_needed = False
        for v in cnts.keys():
            print v, context[v].__class__, cnts[v]
            if not isinstance(context[v], Variable) and v in context.keys() and cnts[v] > 1: 
                lambda_needed = True
        
        # If a lambda is needed, the expression will be a call to a
        # lambda, permitting computation, stored in a variable, to be
        # used in more than one instance.
        if lambda_needed:
            environment.new_frame()
            final_expression = LambdaCall(self.arg_names, rule.expand(environment), called_args)
        # Otherwise, we'll use a tree representation because its simpler.
        else:
            environment.new_frame(context)
            final_expression = rule.expand(environment)

        environment.delete_frame()
        # Return the final expression object.
        return final_expression

class Grammar:
    """
    A series of production rules which define how the space of programs is generated.
    """

    def __init__(self, name):
        """
        Creates a new named grammar.
        """
        self.name = name
        self.prods = {}

    def add_production(self, prod):
        """
        Adds a named production to this grammar.
        """
        pname = prod.get_name()
        if pname in self.prods:
            raise ValueError("Production already exists by the name %s" % pname)
        if pname in self.__dict__:
            raise ValueError("Production has illegal name: %s" % pname)
        self.__dict__[pname] = prod
        self.prods[prod.get_name()] = prod


    def has_production(self, name):
        """
        Returns a boolean indicating whether the grammar has a
        production with the name passed.
        """
        return self.prods.has_key(name)

    def get_production(self, name):
        """
        Returns the production object with the name passed.
        """
        return self.prods.has_key(name)

    def productions(self, prod_name_list):
        """
        Defines several productions by their names with a single call,
        returning a tuple of the production objects created.

        Feature, Morph = grammar.productions(['Feature', 'Morph'])
        """
        L = []
        for arg in prod_name_list:
            prod = Production(arg, self)
            L.append(prod)
        return L


    def __repr__(self):
        """
        Returns a string representation of this grammar object, usually
        in the same syntax as it was parsed.
        """
        return string.join([str(prod) for prod in self.prods.values()], "\n")

class Immediate(RuleExpression):

    def __init__(self, expr):
        """
        Specifies that a Python expression should be evaluated during
        production expansion and its result used as a terminal.
        """
        if not isinstance(expr, RuleExpression):
            expr = LiteralExpression(expr)
        self.expr = expr

    def expand(self, environment):
        expanded = self.expr.expand(environment)
        if not isinstance(expanded, RuleExpression):
            expanded = LiteralExpression(expanded)
        result = expanded.evaluate(environment)
        if not isinstance(result, RuleExpression):
            result = LiteralExpression(result)
        return result

    def count_local_references(self, cnt_dict):
        self.expr.count_local_references(cnt_dict)

    def __repr__(self):
        return "~(%s)" % (self.expr)

def functions(op_name_list, module=None):
    """
    Constructs a tuple of Function objects which can be used in rule
    expressions.

    For example, to create a list of expression objects, do:

        erode, dilate, open, close = functions(['erode', 'dilate', 'open', 'close'])

    Now these functions can be used in rule expressions, e.g.

        Feature[X] = [erode(X)]
    """
    return [Function(arg, module=module) for arg in op_name_list]


def variables(variable_names):
    """
    Defines several Variable objects at once.
    """
    L = []
    for varname in variable_names:
        var = Variable(varname)
        L.append(var)
    return L
