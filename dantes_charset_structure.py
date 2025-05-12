# really dirty script to view the structure of the dantes label charset pkl 

import json

l = list

with open("./dantes_charset.json", mode="r") as fh:
    data= json.load(fh)

lvl = 0

print("k0:", l(data.keys()))

k4_old = None
                
for k1 in data.keys():
    print("k1:", k1, type(data[k1]))
    print("k1:", k1, l(data[k1].keys()))

    for k2 in data[k1].keys():
        print("k1->k2:", f"{k1}->{k2}:\n\t", type(data[k1][k2]))

        if isinstance(data[k1][k2], list):
            print("k1->k2->l:", f"{k1}->{k2}->l:\n\t", set(type(_) for _ in data[k1][k2]))

        elif isinstance(data[k1][k2], dict):
            for k3 in data[k1][k2].keys():
                print("k1->k2->k3:", f"{k1}->{k2}->{k3}:\n\t", type(data[k1][k2][k3]))
                
                for o in data[k1][k2][k3]:
                    assert isinstance(o, dict)
                    k4 = l(o.keys())
                    if k4_old is not None:
                        assert len(k4) == len(k4_old) and all(
                            x==y for (x,y) in zip(sorted(k4),sorted(k4_old))
                        )
                    k4_old = k4
                print("k1->k2->k3->l:", f"{k1}->{k2}->{k3}->l:\n\t", k4)
