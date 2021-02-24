import logging
import json
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from scipy import stats
from astropy import units as u
from lmfit.jsonutils import decode4js, encode4js
from lmfit.printfuncs import params_html_table
from asteval import Interpreter, get_ast_names, valid_symbol_name


logger = logging.getLogger(__name__)


def check_ast_errors(expr_eval):
    """Check for errors derived from asteval."""
    if len(expr_eval.error) > 0:
        expr_eval.raise_exception(None)


class Parameters(OrderedDict):
    """An ordered dictionary of all the Parameter objects required to
    specify a fit model. All minimization and Model fitting routines in
    lmfit will use exactly one Parameters object, typically given as the
    first argument to the objective function.

    All keys of a Parameters() instance must be strings and valid Python
    symbol names, so that the name must match ``[a-z_][a-z0-9_]*`` and
    cannot be a Python reserved word.

    All values of a Parameters() instance must be Parameter objects.

    A Parameters() instance includes an asteval interpreter used for
    evaluation of constrained Parameters.

    Parameters() support copying and pickling, and have methods to convert
    to and from serializations using json strings.

    """

    def __init__(self, usersyms=None, *args, **kwargs):
        """
        Arguments
        ---------
        usersyms : dictionary of symbols to add to the
            :class:`asteval.Interpreter`.
        *args : optional
            Arguments.
        **kwds : optional
            Keyword arguments.

        """
        super().__init__(self)

        self._asteval = Interpreter()

        _syms = {}
        if usersyms is not None:
            _syms.update(usersyms)
        for key, val in _syms.items():
            self._asteval.symtable[key] = val

    def copy(self):
        """Parameters.copy() should always be a deepcopy."""
        return self.__deepcopy__(None)

    def update(self, other):
        """Update values and symbols with another Parameters object."""
        if not isinstance(other, Parameters):
            raise ValueError("'%s' is not a Parameters object" % other)
        self.add_many(*other.values())
        for sym in other._asteval.user_defined_symbols():
            self._asteval.symtable[sym] = other._asteval.symtable[sym]
        return self

    def __copy__(self):
        """Parameters.copy() should always be a deepcopy."""
        return self.__deepcopy__(None)

    def __deepcopy__(self, memo):
        """Implementation of Parameters.deepcopy().

        The method needs to make sure that asteval is available and that all
        individual Parameter objects are copied.

        """
        _pars = Parameters(asteval=None)

        # find the symbols that were added by users, not during construction
        unique_symbols = {key: self._asteval.symtable[key]
                          for key in self._asteval.user_defined_symbols()}
        _pars._asteval.symtable.update(unique_symbols)

        # we're just about to add a lot of Parameter objects to the newly
        parameter_list = []
        for key, par in self.items():
            if isinstance(par, Parameter):
                param = Parameter(name=par.name,
                                  value=par.value,
                                  min=par.min,
                                  max=par.max)
                param.fixed = par.fixed
                param.initials = par.initials
                param.lnprior = par.lnprior
                param.user_data = par.user_data
                parameter_list.append(param)

        _pars.add_many(*parameter_list)

        return _pars

    def __setitem__(self, key, par):
        """Set items of Parameters object."""
        if key not in self:
            if not valid_symbol_name(key):
                raise KeyError("'%s' is not a valid Parameters name" % key)
        if par is not None and not isinstance(par, Parameter):
            raise ValueError("'%s' is not a Parameter" % par)
        OrderedDict.__setitem__(self, key, par)
        par.name = key
        par._eval = self._asteval
        self._asteval.symtable[key] = par.value

    def __add__(self, other):
        """Add Parameters objects."""
        if not isinstance(other, Parameters):
            raise ValueError("'%s' is not a Parameters object" % other)
        out = deepcopy(self)
        out.add_many(*other.values())
        for sym in other._asteval.user_defined_symbols():
            if sym not in out._asteval.symtable:
                out._asteval.symtable[sym] = other._asteval.symtable[sym]
        return out

    def __iadd__(self, other):
        """Add/assign Parameters objects."""
        self.update(other)
        return self

    def __array__(self):
        """Convert Parameters to array."""
        return np.array([float(k) for k in self.values()])

    def __reduce__(self):
        """Reduce Parameters instance such that it can be pickled."""
        # make a list of all the parameters
        params = [self[k] for k in self]

        # find the symbols from _asteval.symtable, that need to be remembered.
        sym_unique = self._asteval.user_defined_symbols()
        unique_symbols = {key: deepcopy(self._asteval.symtable[key])
                          for key in sym_unique}

        return self.__class__, (), {'unique_symbols': unique_symbols,
                                    'params': params}

    def __setstate__(self, state):
        """Unpickle a Parameters instance.

        Parameters
        ----------
        state : dict
            state['unique_symbols'] is a dictionary containing symbols that
            need to be injected into _asteval.symtable
            state['params'] is a list of Parameter instances to be added

        """
        # first update the Interpreter symbol table. This needs to be done
        # first because Parameter's early in the list may depend on later
        # Parameter's. This leads to problems because add_many eventually leads
        # to a Parameter value being retrieved with _getval, which, if the
        # dependent value hasn't already been added to the symtable, leads to
        # an Error. Another way of doing this would be to remove all the expr
        # from the Parameter instances before they get added, then to restore
        # them.

        symtab = self._asteval.symtable
        for key, val in state['unique_symbols'].items():
            if key not in symtab:
                symtab[key] = val

        # then add all the parameters
        self.add_many(*state['params'])

    def eval(self, expr):
        """Evaluate a statement using the asteval Interpreter.

        Parameters
        ----------
        expr : string
            An expression containing parameter names and other symbols
            recognizable by the asteval Interpreter.

        Returns
        -------
           The result of the expression.

        """
        return self._asteval.eval(expr)

    # def update_constraints(self):
    #     """Update all constrained parameters, checking that dependencies are
    #     evaluated as needed."""
    #     requires_update = {name for name, par in self.items() if par._expr is
    #                        not None}
    #     updated_tracker = set(requires_update)
    #
    #     def _update_param(name):
    #         """Update a parameter value, including setting bounds.
    #
    #         For a constrained parameter (one with an `expr` defined),
    #         this first updates (recursively) all parameters on which the
    #         parameter depends (using the 'deps' field).
    #
    #         """
    #         par = self.__getitem__(name)
    #         if par._expr_eval is None:
    #             par._expr_eval = self._asteval
    #         for dep in par._expr_deps:
    #             if dep in updated_tracker:
    #                 _update_param(dep)
    #         self._asteval.symtable[name] = par.value
    #         updated_tracker.discard(name)
    #
    #     for name in requires_update:
    #         _update_param(name)

    def pretty_repr(self, oneline=False):
        """Return a pretty representation of a Parameters class.

        Parameters
        ----------
        oneline : bool, optional
            If True prints a one-line parameters representation (default is
            False).

        Returns
        -------
        s: str
           Parameters representation.

        """
        if oneline:
            return super().__repr__()
        s = "Parameters({\n"
        for key in self.keys():
            s += "    '%s': %s, \n" % (key, self[key])
        s += "    })\n"
        return s

    def pretty_print(self, oneline=False, colwidth=8, precision=4, fmt='g', columns=None):
        """Pretty-print of parameters data.

        Parameters
        ----------
        oneline : bool, optional
            If True prints a one-line parameters representation (default is
            False).
        colwidth : int, optional
            Column width for all columns specified in :attr:`columns`.
        precision : int, optional
            Number of digits to be printed after floating point.
        fmt : {'g', 'e', 'f'}, optional
            Single-character numeric formatter. Valid values are: 'f' floating
            point, 'g' floating point and exponential, or 'e' exponential.
        columns : :obj:`list` of :obj:`str`, optional
            List of :class:`Parameter` attribute names to print.

        """
        if columns is None:
            columns = ['value', 'min', 'max', 'fixed', 'initials', 'lnprior']

        if oneline:
            print(self.pretty_repr(oneline=oneline))
            return

        name_len = max(len(s) for s in self)
        allcols = ['name'] + columns
        title = '{:{name_len}} ' + len(columns) * ' {:>{n}}'
        print(title.format(*allcols, name_len=name_len, n=colwidth).title())
        numstyle = '{%s:>{n}.{p}{f}}'  # format for numeric columns
        otherstyles = dict(name='{name:<{name_len}} ',  fixed='{fixed!s:>{n}}',
                           initials='{initials!s:>{n}}', lnprior='{lnprior!s:>{n}}')
        line = ' '.join([otherstyles.get(k, numstyle % k) for k in allcols])
        for name, values in sorted(self.items()):
            pvalues = {k: getattr(values, k) for k in columns}
            pvalues['name'] = name
            # # stderr is a special case: it is either numeric or None (i.e. str)
            # if 'stderr' in columns and pvalues['stderr'] is not None:
            #     pvalues['stderr'] = (numstyle % '').format(
            #         pvalues['stderr'], n=colwidth, p=precision, f=fmt)
            print(line.format(name_len=name_len, n=colwidth, p=precision,
                              f=fmt, **pvalues))

    def _repr_html_(self):
        """Returns a HTML representation of parameters data."""
        return params_html_table(self)

    def add(self, name, value=None, unit=None, fixed=False, min=-np.inf, max=np.inf, initials=None,
            lnprior=None):
        """Add a Parameter.

        Parameters
        ----------
        name : str
            Name of parameter.  Must match ``[a-z_][a-z0-9_]*`` and cannot be
            a Python reserved word.
        value : float, optional
            Numerical Parameter value, typically the *initial value*.
        unit : str or astropy.Unit, optional
            The physical unit of the parameter
        fixed : bool, optional
            Whether the Parameter is fixed during a fit (default is False).
        min : float, optional
            Lower bound for value (default is `-numpy.inf`, no lower bound).
        max : float, optional
            Upper bound for value (default is `numpy.inf`, no upper bound).
        initials : str, optional
            Mathematical expression used to obtain initial values for a MCMC
            run.
        lnprior : str, optional
            Mathematical expression for calculating the log of the prior
            probability of a paramter value.

        Examples
        --------
        >>> params = Parameters()
        >>> params.add('xvar', value=0.50, min=0, max=1)
        >>> params.add('yvar', expr='1.0 - xvar')

        which is equivalent to:

        >>> params = Parameters()
        >>> params['xvar'] = Parameter(name='xvar', value=0.50, min=0, max=1)
        >>> params['yvar'] = Parameter(name='yvar', expr='1.0 - xvar')

        """
        if isinstance(name, Parameter):
            self.__setitem__(name.name, name)
        else:
            self.__setitem__(name, Parameter(value=value, unit=unit, name=name,
                                             fixed=fixed, min=min, max=max,
                                             initials=initials, lnprior=lnprior))

    def add_many(self, *parlist):
        """Add many parameters, using a sequence of tuples.

        Parameters
        ----------
        parlist : :obj:`sequence` of :obj:`tuple` or :class:`Parameter`
            A sequence of tuples, or a sequence of `Parameter` instances. If
            it is a sequence of tuples, then each tuple must contain at least
            the name. The order in each tuple must be `(name, value, vary,
            min, max, expr, brute_step)`.

        Examples
        --------
        >>>  params = Parameters()
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        >>> params.add_many(('amp', 10, True, None, None, None, None),
        ...                 ('cen', 4, True, 0.0, None, None, None),
        ...                 ('wid', 1, False, None, None, None, None),
        ...                 ('frac', 0.5))
        # add a sequence of Parameters
        >>> f = Parameter('par_f', 100)
        >>> g = Parameter('par_g', 2.)
        >>> params.add_many(f, g)
        """
        __params = []
        for par in parlist:
            if not isinstance(par, Parameter):
                par = Parameter(*par)
            __params.append(par)
            # par._delay_asteval = True
            self.__setitem__(par.name, par)

        # for para in __params:
        #     para._delay_asteval = False

    def valuesdict(self):
        """Return an ordered dictionary of parameter values.

        Returns
        -------
        OrderedDict
           An ordered dictionary of :attr:`name`::attr:`value` pairs for each
           Parameter.

        """
        return OrderedDict((p.name, p.value) for p in self.values())

    def dumps(self, **kws):
        """Represent Parameters as a JSON string.

        Parameters
        ----------
        **kws : optional
            Keyword arguments that are passed to `json.dumps()`.

        Returns
        -------
        str
           JSON string representation of Parameters.

        See Also
        --------
        dump(), loads(), load(), json.dumps()

        """
        params = [p.__getstate__() for p in self.values()]
        sym_unique = self._asteval.user_defined_symbols()
        unique_symbols = {key: encode4js(deepcopy(self._asteval.symtable[key]))
                          for key in sym_unique}
        return json.dumps({'unique_symbols': unique_symbols,
                           'params': params}, **kws)

    def loads(self, s, **kws):
        """Load Parameters from a JSON string.

        Parameters
        ----------
        s : str
            JSON string to load parameters from.
        **kws : optional
            Keyword arguments that are passed to `json.loads()`.

        Returns
        -------
        :class:`Parameters`
           Updated Parameters from the JSON string.

        Notes
        -----
        Current Parameters will be cleared before loading the data from the
        JSON string.

        See Also
        --------
        dump(), dumps(), load(), json.loads()

        """
        self.clear()

        tmp = json.loads(s, **kws)
        unique_symbols = {key: decode4js(tmp['unique_symbols'][key]) for key
                          in tmp['unique_symbols']}

        state = {'unique_symbols': unique_symbols, 'params': []}
        for parstate in tmp['params']:
            _par = Parameter(name='')
            _par.__setstate__(parstate)
            state['params'].append(_par)
        self.__setstate__(state)
        return self

    def dump(self, fp, **kws):
        """Write JSON representation of Parameters to a file-like object.

        Parameters
        ----------
        fp : file-like object
            An open and ``.write()``-supporting file-like object.
        **kws : optional
            Keyword arguments that are passed to `dumps()`.

        Returns
        -------
        int
            Return value from `fp.write()`: the number of characters written.

        See Also
        --------
        dump(), load(), json.dump()

        """
        return fp.write(self.dumps(**kws))

    def load(self, fp, **kws):
        """Load JSON representation of Parameters from a file-like object.

        Parameters
        ----------
        fp : file-like object
            An open and ``.read()``-supporting file-like object.
        **kws : optional
            Keyword arguments that are passed to `loads()`.

        Returns
        -------
        :class:`Parameters`
           Updated Parameters loaded from `fp`.

        See Also
        --------
        dump(), loads(), json.load()

        """
        return self.loads(fp.read(), **kws)


class Parameter(object):

    def __init__(self, name, value=None, unit=None, fixed=False, min=-np.inf, max=np.inf,
                 initials=None, lnprior=None, user_data=None):

        super(Parameter, self).__init__()

        self.name = name
        self.fixed = fixed
        self.min = min
        self.max = max
        self.user_data = user_data
        self._lnprior = lnprior
        self._initials = initials
        self._eval = None
        self._initials_ast = None
        self._lnprior_ast = None
        self._deps = None
        # self._delay_asteval = False  # Not used at the moment

        # Initialize unit prior to value to that value is converted to requested unit if it is Quantity instance
        self.value = None
        self.unit = None

        self._set_unit(unit)
        self._set_value(value)

        self._init_bounds()

    def set(self, value=None, unit=None, fixed=None, min=None, max=None, initials=None, lnprior=None):

        if unit is not None:
            self._set_unit(unit)

        if value is not None:
            self._set_value(value)

        if fixed is not None:
            self.fixed = fixed

        if min is not None:
            self.min = min

        if max is not None:
            self.max = max

        self._init_bounds()

        if initials is not None:
            self.__set_initials(initials)

        if lnprior is not None:
            self.__set_lnprior(lnprior)

    @property
    def initials(self):
        return self._initials

    @initials.setter
    def initials(self, val):
        self.__set_initials(val)

    def __set_initials(self, val):
        if val == '':
            val = None
        self._initials = val
        if val is None:
            self._initials_ast = None
        if val is not None and self._eval is not None:
            self._eval.error = []
            self._eval.error_msg = None
            self._initials_ast = self._eval.parse(val)
            check_ast_errors(self._eval)
            self._deps_initials = get_ast_names(self._initials_ast)

    def evaluate_initials(self, n):
        if self._initials is not None:
            if self._initials_ast is None:
                self.__set_initials(self._initials)
            if self._eval is not None:
                # if not self._delay_asteval:
                fct = self._eval(self._initials_ast)
                check_ast_errors(self._eval)
                return fct(n)
            else:
                raise IOError("Cannot evaluate 'initials' expression: '{0}'".format(self._initials))
        else:
            loc = self.value
            scale = 1
            if self.min == -np.inf and self.max == np.inf:
                fct = stats.norm(loc=loc, scale=scale)
            else:
                fct = stats.truncnorm((self.min - loc) / scale, (self.max - loc) / scale, loc=loc, scale=scale)
            return fct.rvs(n)

    @property
    def lnprior(self):
        return self._lnprior

    @lnprior.setter
    def lnprior(self, val):
        self.__set_lnprior(val)

    def __set_lnprior(self, val):
        if val == '':
            val = None
        self._lnprior = val
        if val is None:
            self._lnprior_ast = None
        if val is not None and self._eval is not None:
            self._eval.error = []
            self._eval.error_msg = None
            self._lnprior_ast = self._eval.parse(val)
            check_ast_errors(self._eval)
            self._deps = get_ast_names(self._lnprior_ast)

    def evaluate_lnprior(self, val):
        if isinstance(val, u.Quantity):
            if self.unit is not None:
                val = val.to(self.unit).value
            else:
                logger.warning("Evaluating prior under assumption that units agree.")
                val = val.value
        if val < self.min or val > self.max:
            return -np.inf
        if self._lnprior is not None:
            if self._lnprior_ast is None:
                self.__set_lnprior(self._lnprior)
            if self._eval is not None:
                # if not self._delay_asteval:
                fct = self._eval(self._lnprior_ast)
                check_ast_errors(self._eval)
                return fct(val)
            else:
                raise IOError("Cannot evaluate expression: '{0}'".format(self._lnprior))
        else:
            return 0

    def _set_value(self, val):

        if isinstance(val, u.Quantity):
            _val = val.value
            _unit = val.unit
            if self.unit is not None:
                try:
                    f = _unit.to(self.unit)
                except u.UnitConversionError:
                    raise IOError(
                        "Unit '{0}' of new value incompatible with existing unit '{1}'.".format(_unit, self.unit))
                else:
                    _val *= f
            else:
                self._set_unit(_unit)
        else:
            _val = val
        self.value = _val

    def _set_unit(self, unit):

        if unit is None:
            return

        _unit = u.Unit(unit)
        if self.unit is None:
            self.unit = _unit
        elif _unit != self.unit:
            logger.error("Cannot change unit from '{0}' to '{1}'.".format(self.unit, _unit))

    def _init_bounds(self):
        """Make sure initial bounds are self-consistent."""
        # _val is None means - infinity.
        if self.max is None:
            self.max = np.inf
        if self.min is None:
            self.min = -np.inf
        if isinstance(self.min, u.Quantity):
            if self.unit is None:
                self.unit = self.min.unit
            try:
                self.min = self.min.to(self.unit).value
            except u.UnitConversionError:
                raise IOError("Incompatible units provided for 'min' of parameter '%s'.".format(self.name))
        if isinstance(self.max, u.Quantity):
            if self.unit is None:
                self.unit = self.max.unit
            try:
                self.max = self.max.to(self.unit).value
            except u.UnitConversionError:
                raise IOError("Incompatible units provided for 'max' of parameter '%s'.".format(self.name))
        if self.value is None:
            if np.isfinite(self.min) & np.isfinite(self.max):
                self.value = (self.min + self.max) / 2.
            else:
                self.value = 0.
        if self.min > self.max:
            self.min, self.max = self.max, self.min
        if np.isclose(self.min, self.max, atol=1e-13, rtol=1e-13):
            raise ValueError("Parameter '%s' has min == max" % self.name)
        if self.value > self.max:
            self.value = self.max
        if self.value < self.min:
            self.value = self.min

    def __repr__(self):
        """Return printable representation of a Parameter object."""
        s = []
        sval = "value=%s" % repr(self.value)
        if self.fixed:
            sval += " (fixed)"
        if self.unit is not None:
            sval += " unit={0}".format(self.unit)
        # elif self.stderr is not None:
        #     sval += " +/- %.3g" % self.stderr
        s.append(sval)
        s.append("bounds=[%s:%s]" % (repr(self.min), repr(self.max)))
        if self._initials is not None:
            s.append("initials='%s'" % self.initials)
        if self._lnprior is not None:
            s.append("lnprior=%s" % self.lnprior)
        return "<Parameter '%s', %s>" % (self.name, ', '.join(s))

    def __getstate__(self):
        """Get state for pickle."""
        return (self.name, self.value, self.unit, self.fixed, self.min,
                self.max, self.initials, self.lnprior, self.user_data)

    def __setstate__(self, state):
        """Set state for pickle."""
        (self.name, _value, _unit, self.fixed, self.min, self.max,
         self._initials, self._lnprior, self.user_data) = state
        self._initials_ast = None
        self._lnprior_ast = None
        self._eval = None
        self._deps = []
        # self._delay_asteval = False
        self.unit = None
        self.value = None
        self._set_unit(unit=_unit)
        self._set_value(val=_value)
        self._init_bounds()
