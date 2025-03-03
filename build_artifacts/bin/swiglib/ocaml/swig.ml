(* -*- tuareg -*- *)
open Int32
open Int64

type enum = [ `Int of int ]

type 'a c_obj_t = 
    C_void
  | C_bool of bool
  | C_char of char
  | C_uchar of char
  | C_short of int
  | C_ushort of int
  | C_int of int
  | C_uint of int32
  | C_int32 of int32
  | C_int64 of int64
  | C_float of float
  | C_double of float
  | C_ptr of int64 * int64
  | C_array of 'a c_obj_t array
  | C_list of 'a c_obj_t list
  | C_obj of (string -> 'a c_obj_t -> 'a c_obj_t)
  | C_string of string
  | C_enum of 'a
  | C_director_core of 'a c_obj_t * 'a c_obj_t option ref

type c_obj = enum c_obj_t

exception BadArgs of string
exception BadMethodName of string * string
exception NotObject of c_obj
exception NotEnumType of c_obj
exception LabelNotFromThisEnum of c_obj
exception InvalidDirectorCall of c_obj
exception NoSuchClass of string
let rec invoke obj = 
  match obj with 
      C_obj o -> o 
    | C_director_core (o,r) -> invoke o
    | _ -> raise (NotObject (Obj.magic obj))
let _ = Callback.register "swig_runmethod" invoke

let fnhelper arg =
  match arg with C_list l -> l | C_void -> [] | _ -> [ arg ]

let director_core_helper fnargs =
  try
    match List.hd fnargs with
      | C_director_core (o,r) -> fnargs
      | _ -> C_void :: fnargs
  with Failure _ -> C_void :: fnargs

let rec get_int x = 
  match x with
      C_bool b -> if b then 1 else 0
    | C_char c
    | C_uchar c -> (int_of_char c)
    | C_short s
    | C_ushort s
    | C_int s -> s
    | C_uint u
    | C_int32 u -> (Int32.to_int u)
    | C_int64 u -> (Int64.to_int u)
    | C_float f -> (int_of_float f)
    | C_double d -> (int_of_float d)
    | C_ptr (p,q) -> (Int64.to_int p)
    | C_obj o -> (try (get_int (o "int" C_void))
		  with _ -> (get_int (o "&" C_void)))
    | _ -> raise (Failure "Can't convert to int")

let rec get_float x = 
  match x with
      C_char c
    | C_uchar c -> (float_of_int (int_of_char c))
    | C_short s -> (float_of_int s)
    | C_ushort s -> (float_of_int s)
    | C_int s -> (float_of_int s)
    | C_uint u
    | C_int32 u -> (float_of_int (Int32.to_int u))
    | C_int64 u -> (float_of_int (Int64.to_int u))
    | C_float f -> f
    | C_double d -> d
    | C_obj o -> (try (get_float (o "float" C_void))
		  with _ -> (get_float (o "double" C_void)))
    | _ -> raise (Failure "Can't convert to float")

let rec get_char x =
  (char_of_int (get_int x))

let rec get_string x = 
  match x with 
      C_string str -> str
    | _ -> raise (Failure "Can't convert to string")

let rec get_bool x = 
  match x with
      C_bool b -> b
    | _ -> 
	(try if get_int x != 0 then true else false
	 with _ -> raise (Failure "Can't convert to bool"))

let disown_object obj = 
  match obj with
      C_director_core (o,r) -> r := None
    | _ -> raise (Failure "Not a director core object")
let _ = Callback.register "caml_obj_disown" disown_object
let addr_of obj = 
  match obj with
      C_obj _ -> (invoke obj) "&" C_void
    | C_director_core (self,r) -> (invoke self) "&" C_void
    | C_ptr _ -> obj
    | _ -> raise (Failure "Not a pointer.")
let _ = Callback.register "caml_obj_ptr" addr_of

let make_float f = C_float f
let make_double f = C_double f
let make_string s = C_string s
let make_bool b = C_bool b
let make_char c = C_char c
let make_char_i c = C_char (char_of_int c)
let make_uchar c = C_uchar c
let make_uchar_i c = C_uchar (char_of_int c)
let make_short i = C_short i
let make_ushort i = C_ushort i
let make_int i = C_int i
let make_uint i = C_uint (Int32.of_int i)
let make_int32 i = C_int32 (Int32.of_int i)
let make_int64 i = C_int64 (Int64.of_int i)

let new_derived_object cfun x_class args =
  begin
    let get_object ob =
      match !ob with
          None ->
    raise (NotObject C_void)
        | Some o -> o in
    let ob_ref = ref None in
    let class_fun class_f ob_r =
      (fun meth args -> class_f (get_object ob_r) meth args) in
    let new_class = class_fun x_class ob_ref in
    let dircore = C_director_core (C_obj new_class,ob_ref) in
    let obj =
    cfun (match args with
            C_list argl -> (C_list ((dircore :: argl)))
	  | C_void -> (C_list [ dircore ])
          | a -> (C_list [ dircore ; a ])) in
    ob_ref := Some obj ;
      obj
  end
  
let swig_current_type_info = ref C_void
let find_type_info obj = !swig_current_type_info 
let _ = Callback.register "swig_find_type_info" find_type_info
let set_type_info obj =
  match obj with
    C_ptr _ -> swig_current_type_info := obj ;
               obj
    | _ -> raise (Failure "Internal error: passed non pointer to set_type_info")
let _ = Callback.register "swig_set_type_info" set_type_info

let class_master_list = Hashtbl.create 20
let register_class_byname nm co = 
  Hashtbl.replace class_master_list nm (Obj.magic co)
let create_class nm =
  try (Obj.magic (Hashtbl.find class_master_list nm)) with _ -> raise (NoSuchClass nm)
