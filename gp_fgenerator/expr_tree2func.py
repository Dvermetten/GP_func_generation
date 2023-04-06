
import sys
import numpy as np


#%%
def tree2func(tree, dict_ephemeral, x):
    exp_ = generate_tree2exp(tree)
    str_f = generate_exp2fun(exp_, dict_ephemeral, x)
    str_code = f'''import numpy as np\ndef func(x):
    return {str_f}
    '''
    code_object = compile(str_code, '<string>', 'exec')
    module = type(sys)('my_module')
    exec(code_object, module.__dict__)
    return module.func
# END DEF
    
#%%
# Convert the tree to the reverse Polish expression
def generate_tree2exp(tree):
    if (tree.get_type() == 0):
        exp = tree.value
    elif (tree.get_type() == 1):
        exp = [generate_tree2exp(tree.left), tree.value]
    elif (tree.get_type() == 2):
        exp = [generate_tree2exp(tree.left), generate_tree2exp(tree.right), tree.value]
    return exp
# END DEF

#%%
# flatten list of lists recursively
def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])
# END DEF

#%%
# % Convert the reverse Polish expression to the function

# %           Meaning                     Syntax
# % 1         Real number in 1-10         1.5
# % 2         Decision vector             (x1,...,xd)
# % 3         First decision variable     x1
# % 4         Translated decision vector  (x2,...,xd,0)
# % 5         Rotated decision vector     XR
# % 6         Index vector                (1,...,d)
# % 7         Random number in 1-1.1      rand() 
# % 11        Addition                    x+y
# % 12        Subtraction                 x-y
# % 13        Multiplication              x.*y
# % 14        Division                    x./y
# % 21        Negative                    -x
# % 22        Reciprocal                  1./x
# % 23        Multiplying by 10           10.*x
# % 24        Square                      x.^2
# % 25        Square root                 sqrt(abs(x))
# % 26        Absolute value              abs(x)
# % 27        Rounded value               round(x)
# % 28        Sine                        sin(2*pi*x)
# % 29        Cosine                      cos(2*pi*x)
# % 30        Logarithm                   log(abs(x))
# % 31        Exponent                    exp(x)
# % 32        sum of vector           	  sum(x)
# % 33        Mean of vector          	  mean(x)
# % 34        Cumulative sum of vector	  cumsum(x)
# % 35        Product of vector           prod(x)
# % 36        Maximum of vector           max(x)
# Convert the reverse Polish expression to the function
def generate_exp2fun(exp, dict_ephemeral, x):
    if (isinstance(exp, int)):
        exp_flat = [exp]
    else:
        exp_flat = flatten(exp)
    str_main = []
    for item in exp_flat:
        if (item < 0):
            str_main.append(f'(abs({item}))')
        else:
            if (item == 1):
                # Real number in 1-10
                # str_main.append(str(np.random.random()*9+1))
                str_ = dict_ephemeral['rand_num'][0]
                str_main.append(f'{str_}')
                dict_ephemeral['rand_num'].pop(0)
            elif (item == 2):
                # Decision vector
                str_main.append('x')    
            elif (item == 3):
                # First decision variable
                str_main.append(f'x[:,0].reshape({len(x)},1)')
            elif (item == 4):
                # Translated decision vector
                str_main.append('(np.vstack((x[:,1:].ravel(), np.zeros((len(x), 1)).ravel())).T)')
            elif (item == 5):
                # Rotated decision vector
                # mat_rand = str(np.random.rand(x.shape[1], x.shape[1]).tolist())
                # str_main.append(f'(np.dot(x, np.array({mat_rand})))')
                str_ = dict_ephemeral['rot_mat'][0]
                str_main.append(f'{str_.tolist()}')
                dict_ephemeral['rot_mat'].pop(0)
            elif (item == 6):
                # Index vector
                ind_ = np.array(range(1, x.shape[1]+1))
                ind_ = str(ind_.reshape(len(ind_), 1).tolist())
                str_main.append(f'(np.array({ind_}))')
            elif (item == 7):
                # Random number in 1-1.1
                # mat_rand = str(np.random.rand(len(x), 1).tolist())
                # str_main.append(f'(1+np.array({mat_rand})/10)')
                str_ = dict_ephemeral['rand_mat'][0]
                str_main.append(f'{str_.tolist()}')
                dict_ephemeral['rand_mat'].pop(0)
            elif (item == 11):
                # Addition
                str_main = str_main[:-2] + [str_main[-2] + '+' + str_main[-1]]
            elif (item == 12):
                # Subtraction
                str_main = str_main[:-2] + [str_main[-2] + '-' + str_main[-1]]
            elif (item == 13):
                # Multiplication
                str_main = str_main[:-2] + [str_main[-2] + '*' + str_main[-1]]
            elif (item == 14):
                # Division
                str_main = str_main[:-2] + [str_main[-2] + '/' + str_main[-1]]
            elif (item == 21):
                # Negative
                str_main = str_main[:-1] + ['-1*(' + str_main[-1] + ')']
            elif (item == 22):
                # Reciprocal
                str_main = str_main[:-1] + ['1/(' + str_main[-1] + ')']
            elif (item == 23):
                # Multiplying by 10
                str_main = str_main[:-1] + ['10*(' + str_main[-1] + ')']
            elif (item == 24):
                # Square
                str_main = str_main[:-1] + ['np.square(' + str_main[-1] + ')']
            elif (item == 25):
                # Square root
                str_main = str_main[:-1] + ['np.sqrt(abs(' + str_main[-1] + '))']
            elif (item == 26):
                # Absolute value
                str_main = str_main[:-1] + ['abs(' + str_main[-1] + ')']
            elif (item == 27):
                # Rounded value
                str_main = str_main[:-1] + ['np.round(' + str_main[-1] + ')']
            elif (item == 28):
                # Sine
                str_main = str_main[:-1] + ['np.sin(2*np.pi*' + str_main[-1] + ')']
            elif (item == 29):
                # Cosine
                str_main = str_main[:-1] + ['np.cos(2*np.pi*' + str_main[-1] + ')']
            elif (item == 30):
                # Logarithm
                str_main = str_main[:-1] + ['np.log(abs(' + str_main[-1] + '))']
            elif (item == 31):
                # Exponent
                str_main = str_main[:-1] + ['np.exp(' + str_main[-1] + ')']
            elif (item == 32):
                # Sum of vector
                str_main = str_main[:-1] + ['np.sum(' + str_main[-1] + f', axis=1).reshape({len(x)}, 1)']
            elif (item == 33):
                # Mean of vector
                str_main = str_main[:-1] + ['np.mean(' + str_main[-1] + f', axis=1).reshape({len(x)}, 1)']
            elif (item == 34):
                # Cumulative sum of vector
                str_main = str_main[:-1] + ['np.cumsum(' + str_main[-1] + f', axis=1).reshape({len(x)}, 1)']
            elif (item == 35):
                # Product of vector
                str_main = str_main[:-1] + ['np.prod(' + str_main[-1] + f', axis=1).reshape({len(x)}, 1)']
            elif (item == 36):
                # Maximum of vector
                str_main = str_main[:-1] + ['np.amax(' + str_main[-1] + f', axis=1).reshape({len(x)}, 1)']
            else:
                raise ValueError(f'Operator {item} is not defined!')
    return str_main[0]
# END DEF