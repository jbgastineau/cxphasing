import MySQLdb
from cxparams import CXParams as CXP
import time
import functools
import array
import sys
import pdb
import cPickle
import re
# Credit here to M. Newville (GSECARS) for SimpleTable and SimpleDB


def clean_input(x, maxlen=None):
    """clean input, forcing it to be a string, with comments stripped,
    and guarding against extra sql statements"""
    if not isinstance(x, (unicode, str)):
        x = str(x)

    if maxlen is None:
        maxlen = 1024
    if len(x) > maxlen:
        x = x[:maxlen-1]
    x.replace('#', '\#')
    eol = x.find(';')
    if eol > -1:
        x = x[:eol]
    return x.strip()

class SimpleDB:

    def __init__(self, user=CXP.db.dbuser, dbname=CXP.db.master_db, passwd=CXP.db.dbpass, host=CXP.db.dbhost,
                 autocommit=1):

        self.conn = MySQLdb.connect(user=user, db=dbname, passwd=passwd, host=host)
        self.cursor = self.conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)

        self.set_autocommit(autocommit)

        self.dbname = dbname

        self.tables = []

        self.read_table_info()

    def __repr__(self):

        return "<SimpleDB name=%s>" % (self.dbname)

    def set_autocommit(self, commit=1):
        self.cursor.execute("set AUTOCOMMIT=%i" % commit)

    def _normalize_dict(self, indict):
        """ internal 'normalization' of query outputs,
        converting unicode to str and array data to lists"""
        t = {}
        if (indict == None):
            return t
        for k, v in indict.items():
            key = k.lower()
            val = v
            if isinstance(v, array.array):
                if v.typecode == 'c':
                    val = v.tostring()
                else:
                    val = v.tolist()
            elif isinstance(v, unicode):
                val = str(v)
            t[key] = val
        return t

    def get_cursor(self, dbconn=None):
        " get a DB cursor, possibly getting a new one from the Connection pool"

        if self.conn is not None:
            if self.cursor is None:
                self.cursor = self.conn.cursor
            return self.cursor

        if self.conn is None:
            CXP.log("Could not start MySQL on %s for database %s" %
                       (self.host, self.dbname),   status='fatal')
            raise IOError("no database connection to  %s" % self.dbname)

    def close(self):
        self.conn.close()

    def use(self, dbname):
        self.execute("use %s" % dbname)

    def read_table_info(self):
        " use database, populate initial list of tables "
        self.table_list = []
        self.tables = {}
        x = self.exec_fetch("show TABLES")

        self.table_list = [i.values()[0] for i in x]
        for i in self.table_list:
            self.tables[i] = SimpleTable(i, db=self)

    def __execute(self, q):
        """internal execution of a single query -- needs a valid cursor!"""

        if self.cursor is None:
            print "SimpleDB.__execute -- no cursor: {}".format(q)
            sys.exit(1)
        n = 0
        while n < 50:
            n = n + 1
            try:
                return self.cursor.execute(q)
            except:
                time.sleep(0.010)
        print "Query Failed: {} ".format(q)
        return None

    def execute(self, q):
        "execute a single sql command string or a tuple or list command strings"
        ret = None
        if isinstance(q, str):
            ret = self.__execute(q)
        elif isinstance(q, (list, tuple)):
            ret = [self.__execute(i) for i in q]
        else:
            self.write("Error: could not execute %s" % str(q))
        return ret

    def exec_fetch(self, q):
        "execute + fetchall"
        self.get_cursor()
        self.__execute(q)
        ret = self.fetchall()
        return ret

    def source_file(self, file=None, report=100):
        """ execute a file of sql commands """
        try:
            f = open(file)
            lines = f.readlines()
            count = 0
            cmds = []
            for x in lines:
                if not x.startswith('#'):
                    x = x[:-1].strip()
                    if x.endswith(';'):
                        cmds.append(x[:-1])
                        sql = "".join(cmds)
                        self.__execute(sql)
                        cmds = []
                    else:
                        cmds.append(x)
                count = count +1
                if (report>0 and (count % report == 0)):
                    print "{:d}} / {:d} ".format(count, len(lines))
            f.close()
        except:
            print "Could not source source_file {}".format(file)

    def create_and_use(self, dbname):
        "create and use a database.  Use with caution!"
        self.__execute("drop database if exists %s" % dbname)
        self.__execute("create database %s" % dbname)
        self.use(dbname)

    def fetchall(self):
        "return all rows from most recent query -- needs valid cursor"
        if self.cursor is None:
            return ()
        r = [self._normalize_dict(i) for i in self.cursor.fetchall()]
        return tuple(r)

    def exec_fetchone(self, q):
        " execute + fetchone"
        self.__execute(q)
        ret = self.fetchone()
        return ret

    def fetchone(self):
        "return next row from most recent query -- needs valid cursor"
        if self.cursor is None:
            return {}
        return self._normalize_dict(self.cursor.fetchone())

    def grant(self, db=None, user=None, passwd=None, host=None, priv=None, grant=False):
        """grant permissions """
        if db is None:
            db = self.dbname
        if user is None:
            user = self.user
        if passwd is None:
            passwd = self.passwd
        if host is None:
            host = self.host
        if priv is None:
            priv = 'all privileges'
        priv = clean_input(priv)
        grant_opt =''
        if grant:
            grant_opt = "with GRANT OPTION"
        cmd = "grant %s on %s.* to %s@%s identified by '%s'  %s"
        self.__execute(cmd % (priv, db, user, host, passwd, grant_opt))

    def sql_exec(self, sql):
        self.execute(sql)


class SimpleTable:
    """ simple MySQL table wrapper class.
    Note: a table must have entry ID"""
    def __init__(self, table=None, db=None):
        self.db = db
        if db is  None:
            CXP.log.error("Warning SimpleTable needs a database connection.")
            return None

        self._name  = None
        if table in self.db.table_list:
            self._name = table
        else:
            table = table.lower()
            if (table in self.db.table_list):
                self._name = table
            else:
                CXP.log.error("Table %s not available in %s " % (table,db))

                return None

        self.fieldtypes = {}
        ret = self.db.exec_fetch("describe %s" % self._name)
        for j in ret:
            field = j['field'].lower()
            vtype = 'str'
            ftype = j['type'].lower()
            if ftype.startswith('int'):     vtype = 'int'
            if ftype.startswith('double'):  vtype = 'double'
            if ftype.startswith('float'):   vtype = 'float'
            if ftype.startswith('blob'):   vtype = 'blob'
            if ftype.startswith('tinyint(1)'): vtype='bool'
            self.fieldtypes[field] = vtype
    
    def update_fieldtypes(self):
            self.fieldtypes = {}
            ret = self.db.exec_fetch("describe %s" % self._name)
            for j in ret:
                field = j['field'].lower()
                vtype = 'str'
                ftype = j['type'].lower()
                if ftype.startswith('int'):     vtype = 'int'
                if ftype.startswith('double'):  vtype = 'double'
                if ftype.startswith('float'):   vtype = 'float'
                if ftype.startswith('blob'):   vtype = 'blob'
                if ftype.startswith('tinyint(1)'): vtype = 'bool'
                self.fieldtypes[field] = vtype

    def check_args(self,**args):
        """ check that the keys of the passed args are all available
        as table columns.
        returns 0 on failure, 1 on success   """
        return self.check_columns(args.keys())

    def check_columns(self,l):
        """ check that the list of args are all available as table columns
        returns 0 on failure, 1 on success
        """
        for i in l:
            if not self.fieldtypes.has_key(i.lower()): return False
        return True
    
    def select_all(self):
        return self.select_where()

    def select_where(self,**args):
        """check for a table row, and return matches"""
        if (self.check_args(**args)):
            q = "select * from %s where 1=1" % (self._name)
            for k,v in args.items():
                k = clean_input(k)
                v = safe_string(v)
                q = "%s and %s=%s" % (q,k,v)
            # print 'S WHERE ', q
            return self.db.exec_fetch(q)
        return 0

    def select(self,vals='*', where='1=1'):
        """check for a table row, and return matches"""
        q= "select %s from %s where %s" % (vals, self._name, where)
        return self.db.exec_fetch(q)

    def select_one(self,vals='*', where='1=1'):
        """check for a table row, and return matches"""
        q= "select %s from %s where %s" % (vals, self._name, where)
        return self.db.exec_fetchone(q)

    def update(self,where='1=1', **kw): # set=None,where=None):
        """update a table row with set and where dictionaries:

           table.update_where({'x':1},{'y':'a'})
        translates to
           update TABLE set x=1 where y=a
           """
        if where==None or set==None:
            self.db.write("update must give 'where' and 'set' arguments")
            return
        try:
            s = []
            for k,v in kw.items():
                if self.fieldtypes.has_key(k):
                    ftype = self.fieldtypes[k]
                    k = clean_input(k)
                    if ftype == 'str':
                        s.append("%s=%s" % (k,safe_string(v)))
                    elif ftype in ('double','float'):
                        s.append("%s=%f" % (k,float(v)))
                    elif ftype == 'int':
                        s.append("%s=%i" % (k,int(v)))
            s = ','.join(s)
            q = "update %s set %s where %s" % (self._name,s,where)
            # print 'UPDATE Q ', q
            self.db.execute(q)
        except:
            self.db.write('update failed: %s' % q)

    def insert(self,**args):
        "add a new table row "
        ok = self.check_args(**args)
        if (ok == 0):
            self.db.write("Bad argument for insert ")
            return 0
        q  = []
        for k,v in args.items():
            if self.fieldtypes.has_key(k):
                ftype = self.fieldtypes[k]
                field = clean_input(k.lower())
                if (v == None): v = ''
                if isinstance(v,(str,unicode)):
                    v = safe_string(v)
                else:
                    v = str(v)
                # v =clean_input(v,maxlen=flen)
                q.append("%s=%s" % (field, v))
        s = ','.join(q)
        qu = "insert into %s set %s" % (self._name, s)
        self.db.execute(qu)

    def insert_on_duplicate_key_update(self, primary={'id': 0}, update={}):
        "insert a row or update a row if a unique key exists"
        q = []
        ks = []
        vs = []
        for k, v in primary.items():
            if self.fieldtypes.has_key(k.lower()):
                ftype = self.fieldtypes[k.lower()]
                field = clean_input(k.lower())
                if ftype in ['int', 'double']:
                    v=str(v)
                elif ftype =='bool':
                    v=str(int(v))
                elif ftype == 'str':
                    v='"{}"'.format(v)
                elif ftype == 'blob':
                    v='"{}"'.format(cPickle.dumps(v))
                ks.append(field)
                vs.append(v)

        for k, v in update.items():
            if self.fieldtypes.has_key(k.lower()):
                ftype = self.fieldtypes[k.lower()]
                field = clean_input(k.lower())
                if ftype in ['int', 'double']:
                    v=str(v)
                elif ftype =='bool':
                    v=str(int(v))
                elif ftype == 'str':
                    v='"{}"'.format(v)
                elif ftype == 'blob':
                    v='"{}"'.format(cPickle.dumps(v))
                ks.append(field)
                vs.append(v)
                q.append("%s=%s" % (field, v))
            else: #new field for database
                if isinstance(v, (int, float)):
                    v=str(v)
                elif isinstance(v, bool):
                    v=str(int(v))
                elif isinstance(v, (str, list, dict)):
                    v='"{}"'.format(v)
                field = clean_input(k.lower())
                ks.append(field)
                vs.append(v)
                q.append("%s=%s" % (field, v))

        s1 = ','.join(ks)
        s2 = ','.join(vs)
        s = ','.join(q)

        qu = 'insert into %s (%s) values (%s) on duplicate key update %s' % (self._name, s1, s2, s)

        self.db.execute(qu)

    def get_new_recon_id(self):
        """
        Get a unique id for the current reconstruction attempt.
        """
        q= "select max(id) as id from slow_params"
        return int(self.db.exec_fetchone(q)['id'])+1

    def db_insert(self):
        def db_insert_decorator(func):
            @functools.wraps(func)
            def db_wrapper(*args, **kwargs):
                #pre
                result = func(args, kwargs)
                #post
                return result
            return db_wrapper
        return db_insert_decorator

    def add_column(self, col_name='col_error', var_type='double', default_value='0'):
        q = 'alter table {} add {} {}'
        if default_value is '':
            def_str = ' default null'
        else:
            def_str = ' default {}'.format(default_value)

        self.db.execute(q.format(self._name, col_name, var_type) + def_str)

    def __repr__(self):
        """ shown when printing instance:
        >>>p = Table()
        >>>print p
        """
        return "<MyTable name=%s>" % (self._name)

    def set_defaults(self):
        """
        Checks table for columns without a default value and gives
        them one.
        """
        self.db.execute('describe {:s}'.format(self._name))

        cols = self.db.fetchall()

        alter = 'alter table {:s} alter {:s} set default {:s}'

        for col in cols:
            if col['default'] is None:
                if re.match('^int|double|tiny*', col['type']):
                    log.warning('Setting default value of {:} to 0'.format(col['field']))
                    self.db.sql_exec(alter.format(self._name, col['field'], str(0)))
                elif re.match('^varchar*', col['type']):
                    log.warning('Setting default value of {:} to ""'.format(col['field']))
                    self.db.sql_exec(alter.format(self._name, col['field'], '""'))
                time.sleep(0.02)
