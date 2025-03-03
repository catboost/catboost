(*Stream:class_ctors*)
let create_$classname_from_ptr raw_ptr =
  C_obj 
begin
  let h = Hashtbl.create 20 in
    List.iter (fun (nm,fn) -> Hashtbl.replace h nm fn) 
	[ "nop", (fun args -> C_void) ;
	  $classbody 
	 "&", (fun args -> raw_ptr) ;
       ":parents",
       (fun args ->
          C_list
	  (let out = ref [] in 
	    Hashtbl.iter (fun x y -> out := (x,y) :: !out) h ;
          (List.map	
	     (fun (x,y) ->
		C_string (String.sub x 2 ((String.length x) - 2)))
	     (List.filter
		(fun (x,y) ->
		   ((String.length x) > 2)
		   && x.[0] == ':' && x.[1] == ':') !out)))) ;
       ":classof", (fun args -> C_string "$realname") ;
       ":methods", (fun args -> 
	  C_list (let out = ref [] in 
	    Hashtbl.iter (fun x y -> out := (C_string x) :: !out) h ; !out))
	] ; 
	let rec invoke_inner raw_ptr mth arg = 
	begin
	  try
	    let application = Hashtbl.find h mth in
	      application
		(match arg with 
		     C_list l -> (C_list (raw_ptr :: l)) 
		   | C_void -> (C_list [ raw_ptr ])
		   | v -> (C_list [ raw_ptr ; v ]))
	  with Not_found -> 
		(* Try parent classes *)
		begin
		  let parent_classes = [
		    $baselist
		  ] in
		  let rec try_parent plist raw_ptr =
		    match plist with
			p :: tl -> 
			  begin
			    try
			      (invoke (p raw_ptr)) mth arg
			    with (BadMethodName (p,m,s)) -> 
			      try_parent tl raw_ptr
			  end
		      | [] ->
			  raise (BadMethodName (raw_ptr,mth,"$realname"))
		  in try_parent parent_classes raw_ptr
		end
	end in
	  (fun mth arg -> invoke_inner raw_ptr mth arg)
end

let _ = register_class_byname "$realname" create_$classname_from_ptr
let _ = Callback.register 
          "create_$normalized_from_ptr"
          create_$classname_from_ptr


(*Stream:mli*)
val create_$classname_from_ptr : c_obj -> c_obj

