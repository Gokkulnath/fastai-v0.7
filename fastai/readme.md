# Understandings :

## imports.py  (Multiple Modules)
Basic Imports/Dependencies required

* in_ipynb: Check whether code is executed in Ipynb
clear_tqdm : Clear progress bar created by tqdm after completion of each epoch


* contextlib : A context manager is an object that defines the runtime context to be established when executing a with statement. The context manager handles the entry into, and the exit from, the desired runtime context for the execution of the block of code. 
Typical uses of context managers include saving and restoring various kinds of global state, locking and unlocking resources, closing opened files, etc.
eg:  with open('file.txt) as f:

* Abstract Base Class:
ABCs introduce virtual subclasses, which are classes that don’t inherit from a class but are still recognized by isinstance() and issubclass();
 
* Duck-typing
A programming style which does not look at an object’s type to determine if it has the right interface; instead, the method or attribute is simply called or used (“If it looks like a duck and quacks like a duck, it must be a duck.”) By emphasizing interfaces rather than specific types, well-designed code improves its flexibility by allowing polymorphic substitution. Duck-typing avoids tests using type() or isinstance(). (Note, however, that duck-typing can be complemented with abstract base classes.) Instead, it typically employs hasattr() tests or EAFP programming.
* EAFP
Easier to ask for forgiveness than permission. This common Python coding style assumes the existence of valid keys or attributes and catches exceptions if the assumption proves false. This clean and fast style is characterized by the presence of many try and except statements. The technique contrasts with the LBYL style common to many other languages such as C.

* from itertools import chain
    * itertools.chain(*iterables)¶
Make an iterator that returns elements from the first iterable until it is exhausted, then proceeds to the next iterable, until all of the iterables are exhausted. Used for treating consecutive sequences as a single sequence. Roughly equivalent to:

        * def chain(*iterables):
            \# chain('ABC', 'DEF') --> A B C D E F
            for it in iterables:
                for element in it:
                    yield element



* from functools import partial:
    * The functools module is for higher-order functions: functions that act on or return other functions.
    * functools.partial(func, *args, **keywords):  
Return a new partial object which when called will behave like func called with the positional arguments args and keyword arguments keywords. If more arguments are supplied to the call, they are appended to args. If additional keyword arguments are supplied, they extend and override keywords.
    * The partial() is used for partial function application which “freezes” some portion of a function’s arguments and/or keywords resulting in a new object with a simplified signature.

* from collections import Iterable, Counter, OrderedDict
    * Iterable - Abstract base class for classes that provide the __iter\__() method
    * Iterator - ABC for classes that provide the __iter\__() and next() methods
    * OrderedDict - Ordered dictionaries are just like regular dictionaries but they remember the order that items were inserted. When iterating over an ordered dictionary, the items are returned in the order their keys were first added.
    * If a new entry overwrites an existing entry, the original insertion position is left unchanged. Deleting an entry and reinserting it will move it to the end.
    * A Counter is a dict subclass for counting hashable objects. It is an unordered collection where elements are stored as dictionary keys and their counts are stored as dictionary values. Counts are allowed to be any integer value including zero or negative counts.

* from operator import itemgetter, attrgetter
    * operator.attrgetter(*attrs):    
Return a callable object that fetches attr from its operand. If more than one attribute is requested, returns a tuple of attributes. The attribute names can also contain dots. For example:   
After f = attrgetter('name'), the call f(b) returns b.name.
After f = attrgetter('name', 'date'), the call f(b) returns (b.name, b.date).
After f = attrgetter('name.first', 'name.last'), the call f(b) returns (b.name.first, b.name.last). 
    * operator.itemgetter(*items):   
Return a callable object that fetches item from its operand using the operand’s __getitem\__() method. If multiple items are specified, returns a tuple of lookup values. For example:   
After f = itemgetter(2), the call f(r) returns r[2].
After g = itemgetter(2, 5, 3), the call g(r) returns (r[2], r[5], r[3]).   


* from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    * The concurrent.futures module provides a high-level interface for asynchronously executing callables. The asynchronous execution can be performed with threads, using ThreadPoolExecutor, or separate processes, using ProcessPoolExecutor. Both implement the same interface, which is defined by the abstract Executor class.
    * The ProcessPoolExecutor class is an Executor subclass that uses a pool of processes to execute calls asynchronously.
    * ThreadPoolExecutor uses a pool of at most max_workers threads to execute calls asynchronously. initializer is an optional callable that is called at the start of each worker thread

## torch_imports.py (torch,torchvision)

* save_model --> Saves the model.state_dict()
* laod_model --> loads the model into suitable location(GPU/CPU). Creates a Set with layer names made up of keys passed as state_dict. the loads the model load_state_dict(). (Prepare the model layout. Create a dict with corresponding keys and then load the weights according to the keys(a.k.a Layer Names)) 
* load_pre   --> Loads the pretrained weights from the weights folder (Hardcoded) (Takes Three Arguments : Pretraine->bool,Architecture-model,filename-string)
* _fastai_model(name, paper_title, paper_href): --> Decorator to dynamically modify the function and update the doc strings 
* Also Has imports to architectures stored in models folder (Custom Architecture defined/torchvision models)


## core.py

* a
* b

## io.py (tqdm,urllib,gzip)

* get_data -- Downloads data from url passed and stores it in the path passed. Has progressbar built inside it.


## set_spawn.py (multiprocessing)

*  multiprocessing supports three ways to start a process. These start methods are:
    *  spawn:  
    The parent process starts a fresh python interpreter process. The child process will only inherit those resources necessary to run the process objects run() method. In particular, unnecessary file descriptors and handles from the parent process will not be inherited. Starting a process using this method is rather slow compared to using fork or forkserver.
    * fork:   The parent process uses os.fork() to fork the Python interpreter. The child process, when it begins, is effectively identical to the parent process. All resources of the parent are inherited by the child process. Note that safely forking a multithreaded process is problematic. Available on Unix only. The default on Unix.
    * forkserver:   When the program starts and selects the forkserver start method, a server process is started. From then on, whenever a new process is needed, the parent process connects to the server and requests that it fork a new process. The fork server process is single threaded so it is safe for it to use os.fork(). No unnecessary resources are inherited.set_start_method() should not be used more than once in the program.



# Dataset.py 

## BaseDataset(Dataset)-> transform
    An abstract class representing a fastai dataset. Extends torch.utils.data.Dataset. 
    
* Variables : Transform, num of items,num of classes and size.


* abstractmethod : 
    - get_n(): Return number of elements in the dataset == len(self)
    - get_c():Return number of classes in a dataset
    - get_sz(): Return maximum size of an image in a dataset.
    - get_x(, i): Return i-th example (image, wav, etc)
    - get_y(, i): Return i-th label

* Properties : 
  - is_reg : True if the data set is used to train regression models
  - is_multi : Returns true if this data set contains multiple labels per sample   


@abstractmethod: A decorator indicating abstract methods.

    Abstract classes are classes that contain one or more abstract methods. An abstract method is a method that is declared, but contains no implementation. Abstract classes may not be instantiated, and require subclasses to provide implementations for the abstract methods. Subclasses of an abstract class in Python are not required to implement abstract methods of the parent class.
    Note : A class that is derived from an abstract class cannot be instantiated unless all of its abstract methods are overridden.An abstract method can have an implementation in the abstract class! Even if they are implemented, designers of subclasses will be forced to override the implementation.abstract method of the parent class can be invoked with super() call mechanism


## FilesDataset(BaseDataset):  
     An Subclass to handle filesbased dataset. Adds:fnames and path by extending the BaseDataset.
     Overrides :
       - get_sz - Gets the size of the data (from transforms)
       - get_x - Fetches the data from the file location(Handles image data)
       - get_n - number of files
     Implements following function : 
       - resize_imgs
       - denorm
  

## FilesArrayDataset(FilesDataset):  
    An Subclass to handle filesbased dataset. Adds:label vector (y) by extending the FilesDataset.
    Overrides :
       - get_c - Returns the number of class based on shape of y (i.e y[1])
       - get_y(,i)) - Returns the label at ith sample

## FilesIndexArrayDataset(FilesArrayDataset):
    An Subclass to handle filesbased dataset which has labels as integers vector. Extends the FilesArrayDataset.
    Overrides :
       - get_c - Returns the number of class based on maximum value of y.

## FilesNhotArrayDataset(FilesArrayDataset):
    An Subclass to handle filesbased dataset which has labels as N hot vectors. Extends the FilesArrayDataset.
    Overrides :
        - is_multi(self):  Sets property to True

## FilesIndexArrayRegressionDataset(FilesArrayDataset):
    An Subclass to handle filesbased dataset which has labels that are continuos values (a.k.a Regression Problem)). Extends the FilesArrayDataset.
    Overrides :
       - is_reg(self):  Sets property to True



## ArraysDataset(BaseDataset):
    An Subclass to handle Array based dataset. Adds: x and y vectors extending the BaseDataset.
    Overrides :
        - get_x(, i): return self.x[i]
        - get_y(, i): return self.y[i]
        - get_n(): return len(self.y)
        - get_sz(): return self.x.shape[1]

## ArraysIndexDataset(ArraysDataset):
    An Subclass to handle Array based dataset which has labels as integers vector. Extends the ArraysDataset.
    Overrides :
        - get_c(self): return int(self.y.max())+1
        - get_y(self, i): return self.y[i]


## ArraysIndexRegressionDataset(ArraysIndexDataset):
    An Subclass to handle Array based dataset which has labels that are continuos values (a.k.a Regression Problem)). Extends the FilesArrayDataset.
    Overrides :
        - is_reg(self): Sets property to True


## ArraysNhotDataset(ArraysDataset):
    An Subclass to handle Array based dataset which has labels as N hot vectors. Extends the FilesArrayDataset.
    Overrides :
    - get_c(self): return self.y.shape[1]
    - is_multi(self): Sets property to True


## ModelData():
    Encapsulates DataLoaders and Datasets for training, validation, test. Base class for fastai *Data classes
    Parameters: path, trn_dl, val_dl, test_dl #dl -> dataloader
    
    @classmethod
    def from_dls(cls, path,trn_dl,val_dl,test_dl=None):
        #trn_dl,val_dl = DataLoader(trn_dl),DataLoader(val_dl)
        #if test_dl: test_dl = DataLoader(test_dl)
        return cls(path, trn_dl, val_dl, test_dl)
    Properties:
    - is_reg(self): return self.trn_ds.is_reg
    - is_multi(self): return self.trn_ds.is_multi
    - trn_ds(self): return self.trn_dl.dataset
    - val_ds(self): return self.val_dl.dataset
    - test_ds(self): return self.test_dl.dataset
    - trn_y(self): return self.trn_ds.y
    - val_y(self): return self.val_ds.y

    
    property:
    
    classmethod : 
    Note:classmethod must have a reference to a class object as the first parameter



# Learner.py
     Combines a ModelData object with a nn.Module object, such that you can train that module.
        data (ModelData): An instance of ModelData.
        models(module): chosen neural architecture for solving a supported problem.
        opt_fn(function): optimizer function, uses SGD with Momentum of .9 if none.
        tmp_name(str): output name of the directory containing temporary files from training process
        models_name(str): output name of the directory containing the trained model
        metrics(list): array of functions for evaluating a desired metric. Eg. accuracy.
        clip(float): gradient clip chosen to limit the change in the gradient to prevent exploding gradients 
