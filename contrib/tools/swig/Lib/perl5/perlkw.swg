/* Warnings for Perl keywords */
#define PERLKW(x) %keywordwarn("'" `x` "' is a perl keyword")  `x`
#define PERLBN(x) %builtinwarn("'" `x` "' conflicts with a built-in name in perl")  "::" `x`


/*

  From  http://www.rocketaware.com/perl/perlfunc/

*/

/* Functions for SCALARs or strings*/
PERLBN(chomp);
PERLBN(chop);
PERLBN(chr);
PERLBN(crypt);
PERLBN(hex);
PERLBN(index);
PERLBN(lc);
PERLBN(lcfirst);
PERLBN(length);
PERLBN(oct);
PERLBN(ord);
PERLBN(pack);
PERLBN(reverse);
PERLBN(rindex);
PERLBN(sprintf);
PERLBN(substr);
PERLBN(uc);
PERLBN(ucfirst);

/* Regular expressions and pattern matching */
PERLBN(m);
PERLBN(pos);
PERLBN(quotemeta);
PERLBN(split);
PERLBN(study);

/* Numeric functions */
PERLBN(abs);
PERLBN(atan2);
PERLBN(cos);
PERLBN(exp);
PERLBN(hex);
PERLBN(int);
PERLBN(log);
PERLBN(oct);
PERLBN(rand);
PERLBN(sin);
PERLBN(sqrt);
PERLBN(srand);


/* Functions for real @ARRAYs*/
PERLBN(pop);
PERLBN(push);
PERLBN(shift);
PERLBN(splice);
PERLBN(unshift);

/* Functions for list data*/ 
PERLBN(grep);
PERLBN(join);
PERLBN(map);
PERLBN(qw);
PERLBN(reverse);
PERLBN(sort);
PERLBN(unpack);


/* Functions for real %HASHes*/
PERLBN(delete);
PERLBN(each);
PERLBN(exists);
PERLBN(keys);
PERLBN(values);


/* Input and output functions*/

PERLBN(binmode);
PERLBN(close);
PERLBN(closedir);
PERLBN(dbmclose);
PERLBN(dbmopen);
PERLBN(die);
PERLBN(eof);
PERLBN(fileno);
PERLBN(flock);
PERLBN(format);
PERLBN(getc);
PERLBN(print);
PERLBN(printf);
PERLBN(read);
PERLBN(readdir);
PERLBN(rewinddir);
PERLBN(seek);
PERLBN(seekdir);
PERLBN(select);
PERLBN(syscall);
PERLBN(sysread);
PERLBN(sysseek);
PERLBN(syswrite);
PERLBN(tell);
PERLBN(telldir);
PERLBN(truncate);
PERLBN(warn);
PERLBN(write);


/* Functions for fixed length data or records*/
PERLBN(pack);
PERLBN(read);
PERLBN(syscall);
PERLBN(sysread);
PERLBN(syswrite);
PERLBN(unpack);
PERLBN(vec);


/* Functions for filehandles, files, or directories */
PERLBN(chdir);
PERLBN(chmod);
PERLBN(chown);
PERLBN(chroot);
PERLBN(fcntl);
PERLBN(glob);
PERLBN(ioctl);
PERLBN(link);
PERLBN(lstat);
PERLBN(mkdir);
PERLBN(open);
PERLBN(opendir);
PERLBN(readlink);
PERLBN(rename);
PERLBN(rmdir);
PERLBN(stat);
PERLBN(symlink);
PERLBN(umask);
PERLBN(unlink);
PERLBN(utime);


/* Keywords related to the control flow of your perl program */
PERLKW(caller);
PERLKW(continue);
PERLKW(die);
PERLKW(do);
PERLKW(dump);
PERLKW(eval);
PERLKW(exit);
PERLKW(goto);
PERLKW(last);
PERLKW(next);
PERLKW(redo);
PERLKW(return);
PERLKW(sub);
PERLKW(wantarray);


/* Keywords related to scoping */
PERLKW(caller);
PERLKW(import);
PERLKW(local);
PERLKW(my);
PERLKW(package);
PERLKW(use);


/* Miscellaneous functions */
PERLBN("defined");
PERLBN(dump);
PERLBN(eval);
PERLBN(formline);
PERLBN(local);
PERLBN(my);
PERLBN(reset);
PERLBN(scalar);
PERLBN(undef);
PERLBN(wantarray);


/* Functions for processes and process groups */
PERLBN(alarm);
PERLBN(exec);
PERLBN(fork);
PERLBN(getpgrp);
PERLBN(getppid);
PERLBN(getpriority);
PERLBN(kill);
PERLBN(pipe);
PERLBN(setpgrp);
PERLBN(setpriority);
PERLBN(sleep);
PERLBN(system);
PERLBN(times);
PERLBN(wait);
PERLBN(waitpid);


/* Keywords related to perl modules */
PERLKW(do);
PERLKW(import);
PERLKW(no);
PERLKW(package);
PERLKW(require);
PERLKW(use);


/* Keywords related to classes and object-orientedness */
PERLKW(bless);
PERLKW(dbmclose);
PERLKW(dbmopen);
PERLKW(package);
PERLKW(ref);
PERLKW(tie);
PERLKW(tied);
PERLKW(untie);
PERLKW(use);

/* Functions new in perl5 */
PERLBN(abs);
PERLBN(bless);
PERLBN(chomp);
PERLBN(chr);
PERLBN(exists);
PERLBN(formline);
PERLBN(glob);
PERLBN(import);
PERLBN(lc);
PERLBN(lcfirst);
PERLBN(map);
PERLBN(my);
PERLBN(no);
PERLBN(prototype);
PERLBN(qx);
PERLBN(qw);
PERLBN(readline);
PERLBN(readpipe);
PERLBN(ref);
PERLBN(sub);
PERLBN(sysopen);
PERLBN(tie);
PERLBN(tied);
PERLBN(uc);
PERLBN(ucfirst);
PERLBN(untie);
PERLBN(use);

#undef PERLKW
#undef PERLBN
