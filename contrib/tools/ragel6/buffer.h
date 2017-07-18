/*
 *  Copyright 2003 Adrian Thurston <thurston@complang.org>
 */

/*  This file is part of Ragel.
 *
 *  Ragel is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 * 
 *  Ragel is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 * 
 *  You should have received a copy of the GNU General Public License
 *  along with Ragel; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA 
 */

#ifndef _BUFFER_H
#define _BUFFER_H

#define BUFFER_INITIAL_SIZE 4096

/* An automatically grown buffer for collecting tokens. Always reuses space;
 * never down resizes. */
struct Buffer
{
	Buffer()
	{
		data = (char*) malloc( BUFFER_INITIAL_SIZE );
		allocated = BUFFER_INITIAL_SIZE;
		length = 0;
	}
	~Buffer() { free(data); }

	void append( char p )
	{
		if ( length == allocated ) {
			allocated *= 2;
			data = (char*) realloc( data, allocated );
		}
		data[length++] = p;
	}
		
	void clear() { length = 0; }

	char *data;
	int allocated;
	int length;
};

#endif
