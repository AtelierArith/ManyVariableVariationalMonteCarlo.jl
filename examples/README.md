# Usage

Just run:

```sh
$ julia --project=@. stdface_spin_chain_from_file.jl StdFace.def
```

This should output many files such as `.def` files and `.dat`

```
$ tree output
output
├── coulombinter.def
├── exchange.def
├── greenone.def
├── greentwo.def
├── gutzwilleridx.def
├── hund.def
├── jastrowidx.def
├── locspn.def
├── modpara.def
├── namelist.def
├── orbitalidx.def
├── qptransidx.def
├── trans.def
├── zqp_gutzwiller_opt.dat
├── zqp_jastrow_opt.dat
├── zqp_opt.dat
├── zqp_orbital_opt.dat
├── zvo_CalcTimer.dat
├── zvo_out_001.dat
├── zvo_SRinfo.dat
├── zvo_time_001.dat
└── zvo_var_001.dat

1 directory, 22 files
```
