openshell sandbox list | awk 'NR>1 && $1 != "NAME" {print $1}' | xargs -r openshell sandbox delete
