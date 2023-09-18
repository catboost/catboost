/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * doxyentity.h
 *
 * Part of the Doxygen comment translation module of SWIG.
 * ----------------------------------------------------------------------------- */

#ifndef SWIG_DOXYENTITY_H
#define SWIG_DOXYENTITY_H

#include <string>
#include <list>


class DoxygenEntity;

typedef std::list<DoxygenEntity> DoxygenEntityList;
typedef DoxygenEntityList::iterator DoxygenEntityListIt;
typedef DoxygenEntityList::const_iterator DoxygenEntityListCIt;


/*
 * Structure to represent a doxygen comment entry
 */
class DoxygenEntity {
public:
  std::string typeOfEntity;
  std::string data;
  bool isLeaf;
  DoxygenEntityList entityList;

  DoxygenEntity(const std::string &typeEnt);
  DoxygenEntity(const std::string &typeEnt, const std::string &param1);
  DoxygenEntity(const std::string &typeEnt, const DoxygenEntityList &entList);

  void printEntity(int level) const;
};

#endif
