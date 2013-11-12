#!/usr/bin/env python
#
# creates initial MySQL databases for CXPhasing
# Author: M. Newville
# Modified: D.J. Vine

import os
import sys
import time
import getpass
import warnings
import pdb
import cxphasing
import cxphasing.cxparams.CXParams as CXP
from cxphasing.CXDb import SimpleDB

welcome_msg = """
 **************************************************************
 Initializing the MySQL database tables for the CXPhasing 

 CXPhasing will use mysql host, username, and password:
    host = {dbhost}
    user = {dbuser}
    pass = {dbpass}

 You are about to be prompted for a username / password        
 of a mysql account that can grant permissions to {dbuser} 

 *** Warning *** Warning *** Warning *** Warning *** Warning *** 
 This will destroy the existing databases {dbname}

 Use Ctrl-C now if you do not want these databases destroyed!
 **************************************************************
"""

init_sql = './init_cxdb.sql'


def initialize():
    print welcome_msg.format(** CXP.db.__dict__ )
    
    warnings.filterwarnings("ignore", "Unknown table.*")

    super_user = getpass.getuser()
    super_pass = None

    try:
        answer = raw_input('mysql username [%s]:' % super_user)
    except KeyboardInterrupt:
        print 'exiting...'
        sys.exit(2)
    if answer is not '':  super_user = answer

    try:
        super_pass  = getpass.getpass('mysql password for %s:' % super_user)
    except KeyboardInterrupt:
        print 'exiting...'
        sys.exit(2)


    try:
        xdb   = SimpleDB(user=super_user, dbname=CXP.db.dbname,
                         passwd=super_pass, host = CXP.db.dbhost)
    except:
        raise
        print 'error starting mysql. Invalid Username/Password? Is mysql running?'
        sys.exit(1)


    grant_kws = {'user': super_user,
                 'passwd': super_pass,
                 'host': CXP.db.dbhost,
                 'grant': True}


    print 'creating database %s :' % CXP.db.dbname,
    xdb.create_and_use(CXP.db.dbname)
    xdb.grant(priv='create', db='*', **grant_kws)
    xdb.grant(priv='drop',   db='*', **grant_kws)
    xdb.execute('flush privileges')
    xdb.source_file(init_sql)

    print 'granting permissions...', CXP.db.dbname
    
    xdb.grant(db=CXP.db.dbname, **grant_kws)
    xdb.grant(priv='create', db=CXP.db.dbname, **grant_kws)
    xdb.grant(priv='insert', db=CXP.db.dbname, **grant_kws)
    xdb.grant(priv='alter',  db=CXP.db.dbname, **grant_kws)
    xdb.grant(priv='delete', db=CXP.db.dbname, **grant_kws)
    xdb.grant(priv='all',    db=CXP.db.dbname, **grant_kws)
    xdb.grant(priv='drop',   db=CXP.db.dbname, **grant_kws)
    xdb.execute('flush privileges')

    print 'Done.'

initialize()
