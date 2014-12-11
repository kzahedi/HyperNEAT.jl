module HyperNEAT

using IniFile

export CPPN

type Configuration
  prob_synapse_change_weight::Float64 # 1 - prob_synapse_change_weight = new random value
  prob_synapse_new::Float64
  prob_neuron_new::Float64
  prob_neuron_change_function::Float64
  #= prob_neuron_change_parameter::Float64 =#
  prob_deactivate_synapse::Float64
  prob_activate_synapse::Float64
  value_new_synapse::Float64
  delta_weight::Float64
  #= delta_neuron_parameter::Float64 =#
  c1::Float64
  c2::Float64
  c3::Float64
  c4::Float64
  r::Float64 # mathing percentage
  threshold::Float64 # speciation threshold
  innovation::Int64 # global innovation number
  id::Int64
  life_time::Int64
  max_individuals::Int64
  min_individuals::Int64
  yars_port::Int64
  log_dir::String
  generations::Int64
  population_size::Int64
  use_gui::Bool
  use_single_gui::Bool
  use_plot::Bool
  xml::String
  ports::Vector{Int64}
  wd::String
  structure::String
  normalise_population_size::Bool
  max_weight::Float64
end


function read_cfg(filename::String)
  ini = Inifile()
  read(ini, filename)
  pscw = float(get(ini, "Mutation",     "probability synapse change weight",   -1.0))
  psn  = float(get(ini, "Mutation",     "probability synapse new",             -1.0))
  pnn  = float(get(ini, "Mutation",     "probability neuron new",              -1.0))
  pnc  = float(get(ini, "Mutation",     "probability neuron change function",  -1.0))
  #= pcp  = float(get(ini, "Mutation",     "probability neuron change parameter", -1.0)) =#
  pds  = float(get(ini, "Mutation",     "probability deactivate synapse",      -1.0))
  pas  = float(get(ini, "Mutation",     "probability activate synapse",        -1.0))
  vns  = float(get(ini, "Mutation",     "value new synapse",                   -1.0))
  dw   = float(get(ini, "Mutation",     "delta weight",                        -1.0))
  #= dnp  = float(get(ini, "Mutation",     "neuron parameter delta",              -1.0)) =#
  msw  = float(get(ini, "Mutation",     "maximal synaptic weight",             10.0))
  c1   = float(get(ini, "Speciation",   "c1",                                  -1.0))
  c2   = float(get(ini, "Speciation",   "c2",                                  -1.0))
  c3   = float(get(ini, "Speciation",   "c3",                                  -1.0))
  c4   = float(get(ini, "Speciation",   "c4",                                  -1.0))
  th   = float(get(ini, "Speciation",   "threshold",                           -1.0))
  r    = float(get(ini, "Reproduction", "r",                                   -1.0))
  mai  = int(get(ini,   "Reproduction", "maximum number of individuals",        10))
  mii  = int(get(ini,   "Reproduction", "minimum number of individuals",        10))
  np   = (get(ini,      "Reproduction", "normalise population size", "false") == "true")
  lt   = int(get(ini,   "General",      "life time",                           -1.0))
  yp   = int(get(ini,   "General",      "yars port",                           -1.0))
  ld   = get(ini,       "General",      "log directory",                       ".")
  g    = int(get(ini,   "General",      "generations",                         -1))
  ps   = int(get(ini,   "General",      "population size",                     -1))
  p    = int(get(ini,   "General",      "port",                                []))
  ug   = (get(ini,      "General",      "use gui",  "false")          == "true")
  usg  = (get(ini,      "General",      "use only one gui",  "false") == "true")
  up   = (get(ini,      "General",      "use plot", "false")          == "true")
  xml  = get(ini,       "General",      "xml",                    "")
  wd   = get(ini,       "General",      "working directory",      "")
  st   = get(ini,       "General",      "structure", "neat")

  #ffc  = get(ini, "fitness function coefficients", "c",                                  "-1.0")
  cfg = Configuration(
  pscw,
  psn,
  pnn,
  pnc,
  #= pcp, =#
  pds,
  pas,
  vns,
  dw,
  #= dnp, =#
  c1,
  c2,
  c3,
  c4,
  r,
  th,
  10, # innovation number
  0, # global id
  lt,
  mai,
  mii,
  yp,
  ld,
  g,
  ps,
  ug,
  usg,
  up,
  xml,
  [], # todo ports
  wd,
  st,
  np,
  msw
  )

  return cfg
end

#= function Base.show(io::IO, cfg::Configuration) =#
#= println("[Mutation]") =#
#= println("probability synapse change weight  = ", cfg.prob_synapse_change_weight) =#
#= println("probability synapse new            = ", cfg.prob_synapse_new) =#
#= println("probability neuron new             = ", cfg.prob_neuron_new) =#
#= println("probability neuron change function = ", cfg.prob_neuron_change_function) =#
#= println("probability deactivate synapse     = ", cfg.prob_deactivate_synapse) =#
#= println("probability activate synapse       = ", cfg.prob_activate_synapse) =#
#= println("value new synapse                  = ", cfg.value_new_synapse) =#
#= println("delta weight                       = ", cfg.delta_weight) =#
#= println("") =#
#= println("[Speciation]") =#
#= println("c1                                 = ", cfg.c1) =#
#= println("c2                                 = ", cfg.c2) =#
#= println("c3                                 = ", cfg.c3) =#
#= println("c4                                 = ", cfg.c4) =#
#= println("threshold                          = ", cfg.threshold) =#
#= println("") =#
#= println("[Reproduction]") =#
#= println("r                                  = ", cfg.r) =#
#= println("") =#
#= end =#

function write_cfg(filename::ASCIIString, cfg::Configuration)
  ini = Inifile()
  ini.sections["Mutation"]     = IniFile.HTSS()
  ini.sections["Speciation"]   = IniFile.HTSS()
  ini.sections["Reproduction"] = IniFile.HTSS()

  sec                                       = ini.sections["Mutation"]
  sec["probability synapse change weight"]  = "$(cfg.prob_synapse_change_weight)"
  sec["probability synapse new"]            = "$(cfg.prob_synapse_new)"
  sec["probability neuron new"]             = "$(cfg.prob_neuron_new)"
  sec["probability neuron change function"] = "$(cfg.prob_neuron_change_function)"
  sec["probability deactivate synapse"]     = "$(cfg.prob_deactivate_synapse)"
  sec["probability activate synapse"]       = "$(cfg.prob_activate_synapse)"
  sec["value new synapse"]                  = "$(cfg.value_new_synapse)"
  sec["delta weight"]                       = "$(cfg.delta_weight)"

  sec                                       = ini.sections["Speciation"]
  sec["c1"]                                 = "$(cfg.c1)"
  sec["c2"]                                 = "$(cfg.c2)"
  sec["c3"]                                 = "$(cfg.c3)"
  sec["c4"]                                 = "$(cfg.c4)"
  sec["threshold"]                          = "$(cfg.threshold)"
  sec                                       = ini.sections["Reproduction"]
  sec["r"]                                  = "$(cfg.r)"

  fd = open(filename, "w")
  write(fd, ini)
  close(fd)
end

#c = read_cfg("cfg.ini")
#write_cfg("/Users/zahedi/Desktop/test.ini", c)

type HyperNEATError <: Exception
  var::String
end

type NeuronGene
  innovation::Int64
  ntype::Symbol # 1 = Input Neuron, 2 = Output Neuron, 3 = Hidden Neuron
  func::Symbol
  #= parameters::Vector{Float64} =#
end

#= new_neuron(;innovation=-1, ntype=:input, func=:id, parameters = [1.0, 0.0]) = NeuronGene(innovation, ntype, func, parameters) =#

new_neuron(;innovation=-1, ntype=:input, func=:id) = NeuronGene(innovation, ntype, func)

type SynapseGene
  src::Int64
  dest::Int64
  weight::Float64
  innovation::Int64
  active::Bool
end

new_synapse(;src=-1, dest=-1, weight=0.0, innovation=-1, active=true) = SynapseGene(src, dest, weight, innovation, active)


type Genome
  neurons::Vector{NeuronGene}
  synapses::Vector{SynapseGene}
end

new_genome(;neurons=[], synapses=[]) = Genome(neurons, synapses)


type Individual
  genomes::Vector{Genome}
  fitness::Float64
  id::Int64
  parents::(Int64, Int64)
  species_id::Int64
  age::Int64
end

new_individual(;genomes=[], fitness=0.0, id=-1, parents=(-1,-1), species_id = -1, age = 0) = Individual(genomes, fitness, id, parents, species_id, age)


type Species
  individuals::Vector{Individual}
  size::Int64
end

new_species(;individuals=[], size=-1) = Species(individuals, size)


type Population
  species::Vector{Species}
  generation::Int64
end

new_population(;species=[], generation=-1) = Population(species, generation)


function population_write(filename::String, p::Population)
  f = open(filename, "w")
  serialize(f, p)
  close(f)
end

function population_read(filename::String)
  f = open(filename, "r")
  p = deserialize(f)
  close(f)
  return p
end


function mutation_add_neuron!(genome::Genome, cfg::Configuration)
  if length(genome.synapses) == 0
    return
  end
  if rand() < cfg.prob_neuron_new
    active_synapses = filter(s->s.active, genome.synapses)
    synapse         = active_synapses[ceil(rand() * length(active_synapses))]
    synapse.active  = false
    function_keys   = collect(keys(functions))
    func            = function_keys[ceil(rand() * length(function_keys))]
    cfg.innovation  = cfg.innovation + 1
    genome.neurons  = [genome.neurons, NeuronGene(cfg.innovation, :hidden, func, [1.0, 0.0])]
    neuron_index    = length(genome.neurons)
    cfg.innovation  = cfg.innovation + 1
    genome.synapses = [genome.synapses, SynapseGene(neuron_index, synapse.dest, synapse.weight, cfg.innovation, true)]
    cfg.innovation  = cfg.innovation + 1
    genome.synapses = [genome.synapses, SynapseGene(synapse.src,  neuron_index, 1.0,            cfg.innovation, true)]
  end
end

function mutation_change_neuron!(genome::Genome, cfg::Configuration)
  if rand() < cfg.prob_neuron_change_function
    neurons       = filter(n->(n.ntype == :hidden), genome.neurons)
    if length(neurons) == 0
      return
    end
    neuron        = neurons[int(ceil(rand() * length(neurons)))]
    function_keys = collect(keys(functions))
    func          = neuron.func
    while func == neuron.func
      func          = function_keys[ceil(rand() * length(function_keys))]
    end
    neuron.func   = func
  end
end

#= function mutation_change_neuron_parameter!(genome::Genome, cfg::Configuration) =#
  #= if rand() < cfg.prob_neuron_change_parameter =#
    #= neurons           = filter(n->(n.ntype == :hidden || n.ntype == :output), genome.neurons) =#
    #= neuron            = neurons[int(ceil(rand() * length(neurons)))] =#
    #= neuron.parameters = map(p->p + (2.0 * rand() - 1.0) * cfg.delta_neuron_parameter, neuron.parameters) =#
  #= end =#
#= end =#

function list_of_possible_synapses(genome::Genome)
  last_input_neuron       = length(filter(n->n.ntype == :input, genome.neurons))
  nr_of_non_input_neurons = length(filter(n->n.ntype != :input, genome.neurons))

  all_possible_synapses = []
  for src = 1:length(genome.neurons)
    for dest = last_input_neuron+1:last_input_neuron+nr_of_non_input_neurons
      if length(filter(s->(s.src == src && s.dest == dest), genome.synapses)) == 0
        all_possible_synapses = [all_possible_synapses, (src, dest)]
      end
    end
  end
  all_possible_synapses
end

function mutation_add_synapse!(genome::Genome, cfg::Configuration)
  if rand() < cfg.prob_synapse_new
    l = list_of_possible_synapses(genome)
    if length(l) == 0
      return
    end
    cfg.innovation = cfg.innovation + 1
    (src,dest) = l[int(ceil(rand() * length(l)))]
    genome.synapses = [genome.synapses, SynapseGene(src, dest, (2.0 * rand() - 1.0) * cfg.value_new_synapse, cfg.innovation, true)]
  end
end

function mutation_change_synapse!(genome::Genome, cfg::Configuration)
  if length(genome.synapses) == 0
    return
  end
  synapse_index = int(ceil(rand() * length(genome.synapses)))
  synapse = genome.synapses[synapse_index]
  if rand() < cfg.prob_synapse_change_weight
    synapse.weight = synapse.weight + (2.0 * rand() - 1.0) * cfg.delta_weight
  else
    synapse.weight =                  (2.0 * rand() - 1.0) * cfg.value_new_synapse
  end

  genome.synapses[synapse_index] = synapse
end

function mutation_deactivate_synapse!(genome::Genome, cfg::Configuration)
  if length(genome.synapses) == 0
    return
  end
  if rand() < cfg.prob_deactivate_synapse
    if length(genome.synapses) != sum(map(s->s.active?0:1, genome.synapses))
      return # there no active synapses
    end

    active_synapses = filter(s->s.active, genome.synapse)
    active_synapses[ceil(rand() * length(active_synapses))].active = false

  end
end

function mutation_activate_synapse!(genome::Genome, cfg::Configuration)
  if length(genome.synapses) == 0
    return
  end
  if rand() < cfg.prob_activate_synapse
    if length(genome.synapses) != sum(map(s->s.active?1:0, genome.synapses))
      return # there no deactivated synapses
    end
    deactivated_synapses = filter(s->s.active == false, genome.synapses)
    if length(deactivated_synapses) == 0
      return
    end
    deactivated_synapses[ceil(rand() * length(deactivated_synapses))].active = true
  end
end

function crossover(mother::Genome, father::Genome)
  child = new_genome()
  max_innovation_number = 1
  if length(mother.synapses) > 0 && length(father.synapses) > 0
    max_innovation_number = maximum([map(s->s.innovation, mother.synapses),map(s->s.innovation, father.synapses)])
  elseif length(mother.synapses) > 0
    max_innovation_number = maximum([map(s->s.innovation, mother.synapses)])
  elseif length(father.synapses) > 0
    max_innovation_number = maximum([map(s->s.innovation, father.synapses)])
  end
  for i = 1:max_innovation_number
    # synapses
    ms = filter(s->s.innovation == i, mother.synapses)
    fs = filter(s->s.innovation == i, father.synapses)
    cs = []
    if length(ms) > 0 && length(fs) > 0 && length(ms) != length(fs)
      throw(HyperNEATError("crossover: Number of synapses don't match $(length(ms)) vs $(length(fs)) for innovation $i"))
    end
    if length(ms) > 0 && length(fs) > 0
      cs = [rand() < 0.5?ms[i]:fs[i] for i=1:length(ms)]
    elseif length(ms) > 0
      cs = ms
    else
      cs = fs
    end

    for s in cs
      child.synapses = [child.synapses, SynapseGene(s.src, s.dest, s.weight, s.innovation, s.active)]
    end

    # neurons
    mn = filter(n->n.innovation == i, mother.neurons)
    fn = filter(n->n.innovation == i, father.neurons)
    cn = []
    if (length(mn) > 0 && length(fn) > 0) && (length(mn) != length(fn))
      throw(HyperNEATError("crossover: Number of neurons don't match: father: $(length(fn)) vs mother: $(length(mn))"))
    end
    if length(mn) > 0 && length(fn) > 0
      cn = [rand() < 0.5? mn[i] : fn[i] for i = 1:length(mn)]
    elseif length(mn) > 0
      cn = mn
    else
      cn = fn
    end

    for n in cn
      #= child.neurons = [child.neurons, NeuronGene(n.innovation, n.ntype, n.func, n.parameters)] =#
      child.neurons = [child.neurons, NeuronGene(n.innovation, n.ntype, n.func)]
    end
  end
  child
end


function speciation_distance(g1::Genome, g2::Genome, cfg::Configuration)
  N                        = float(maximum([length(g1.synapses), length(g2.synapses)]))
  max_innovation_number_g1 = length(g1.synapses)>0?maximum(map(s->s.innovation, g1.synapses)):0
  max_innovation_number_g2 = length(g2.synapses)>0?maximum(map(s->s.innovation, g2.synapses)):0
  excess                   = float(abs(max_innovation_number_g1 - max_innovation_number_g2))
  disjoint                 = 0.0
  matching                 = 0.0
  weight_difference        = 0.0
  for i = 1:maximum([max_innovation_number_g1, max_innovation_number_g2])
    found_g1 = filter(s->s.innovation == i, g1.synapses)
    found_g2 = filter(s->s.innovation == i, g2.synapses)
    if length(found_g1) != length(found_g2)
      disjoint = disjoint + 1.0
    elseif length(found_g1) > 0
      weight_difference = weight_difference + abs(found_g1[1].weight - found_g2[1].weight)
      matching = matching + 1.0
    end
  end

  neuron_difference = 0.0
  n_matching        = 0.0
  for i = 1:maximum([max_innovation_number_g1, max_innovation_number_g2])
    found_g1 = filter(n->n.innovation == i, g1.neurons)
    found_g2 = filter(n->n.innovation == i, g2.neurons)
    if length(found_g1) > 0 && length(found_g2) > 0
      if found_g1[1].func != found_g2[1].func
        neuron_difference = neuron_difference + 1.0
        n_matching        = n_matching + 1.0
      end
    else
      neuron_difference = neuron_difference + 1.0
      n_matching        = n_matching + 1.0
    end
  end
  a = (N>0)?((cfg.c1 * excess + cfg.c2 * disjoint) / N):0
  b = (matching>0)?(cfg.c3 * weight_difference / matching):0
  c = (n_matching>0)?(cfg.c4 * neuron_difference / n_matching):0

  a + b + c
end

function calculate_species_sizes!(population::Population, cfg::Configuration)

  fitness_values = []
  for s in population.species
    for individual in s.individuals
      fitness_values = [fitness_values, individual.fitness]
    end
  end

  minimal_fitness = minimum(fitness_values)

  fitness_values = []
  for s in population.species
    l = float(length(s.individuals))
    for individual in s.individuals
      individual.fitness = (individual.fitness + abs(minimal_fitness)) / l
      fitness_values = [fitness_values, individual.fitness]
    end
  end

  average_fitness = mean(fitness_values)

  for s in population.species # new size depends
    ss     = float(sum(map(i->i.fitness, s.individuals)))
    av     = float(average_fitness)
    s.size = maximum([cfg.max_individuals, int(round(sum(map(i->i.fitness, s.individuals)) / average_fitness))])
  end

  for s in population.species
    if s.size > cfg.max_individuals
      s.size = cfg.max_individuals
    end
  end

end

function mutation_scale_weights!(genome::Genome, cfg::Configuration)
  for s in genome.synapses
    if abs(s.weight) > cfg.max_weight
      s.weight = (s.weight<0:-1:1) * cfg.max_weight
    end
  end
end

function mutate!(g::Genome, cfg::Configuration)
  mutation_deactivate_synapse!(g, cfg)
  mutation_add_neuron!(g, cfg)
  #= mutation_change_neuron_parameter!(g, cfg) =#
  mutation_change_synapse!(g, cfg)
  mutation_add_synapse!(g, cfg)
  mutation_change_neuron!(g, cfg)
  mutation_activate_synapse!(g, cfg)
  mutation_scale_weights!(g, cfg)
  g
end


function reproduce!(population::Population, cfg::Configuration)
  for s in population.species
    s.individuals = sort(s.individuals, by=x->x.fitness, rev=true)
  end

  calculate_species_sizes!(population, cfg)

  if cfg.normalise_population_size == true
    sum_sizes = sum(map(s->s.size, population.species))

    for s in population.species
      s.size = int(round(float(s.size) / float(sum_sizes) * cfg.max_individuals))
    end

    if length(population.species) == 1
      if population.species[1].size != cfg.max_individuals
        population.species[1].size = cfg.max_individuals
      end
    end
  else
    sum_sizes = sum(map(s->s.size, population.species))
    if sum_sizes < cfg.min_individuals
      factor = float64(cfg.min_individuals / sum_sizes)
      for s in population.species
        s.size = int(ceil(s.size * factor))
      end
    end
  end

  individuals = []

  for s in population.species
    N               = ceil(cfg.r*length(s.individuals))
    parents         = s.individuals[1:N]
    s.individuals   = filter(p->p.age < 5, parents)
    nr_of_offspring = s.size - length(s.individuals)
    for i in s.individuals
      i.age = i.age + 1
    end
    if s.size > length(s.individuals)
      for i = 1:nr_of_offspring
        mother        = parents[ceil(rand() * N)]
        father        = parents[ceil(rand() * N)]

        nr_of_genomes = length(mother.genomes)

        child         = new_individual()
        child.genomes = map(i->crossover(mother.genomes[i], father.genomes[i]), [1:nr_of_genomes])
        child.parents = (mother.id, father.id)
        child.id      = cfg.id
        cfg.id        = cfg.id + 1

        for g in child.genomes
          mutate!(g, cfg)
        end
        individuals = [individuals, child]
      end
    else
      s.size = 0
    end
  end

  population.generation = population.generation + 1

  population.species = filter(s->s.size > 0, population.species)

  first = new_species(individuals=[individuals[1]])
  population.species = [first]

  for i in individuals[2:end]
    found = false
    for s in population.species
      random_individual = s.individuals[int(ceil(rand() * length(s.individuals)))]
      dist = maximum([speciation_distance(i.genomes[j], random_individual.genomes[j], cfg) for j=1:length(i.genomes)])
      if dist < cfg.threshold
        s.individuals = [s.individuals, i]
        found = true
        break
      end
    end
    if found == false
      population.species = [population.species, new_species(individuals=[i])]
    end
  end
  population
end

type Neuron
  inputs::Vector{Neuron}
  weights::Vector{Float64}
  func::Function
  output::Float64
  #= parameters::Vector{Float64} =#
end

type CPPN
  inputs::Vector{Neuron}
  hidden::Vector{Neuron}
  outputs::Vector{Neuron}
  neurons::Vector{Neuron}
end

function update_neuron!(n)
  act = sum([n.inputs[i].output * n.weights[i] for i=1:length(n.inputs)])
  if act == NaN
    println("Inputs: ",  [n.inputs[i].output for i=1:length(n.inputs)])
    println("Weights: ", [n.weights[i]       for i=1:length(n.inputs)])
  end
  if abs(act) > 1000
    act = (act<0)?-1000.0:1000.0
  end
  #= println(n.parameters) =#
  #= n.output = n.func(n.parameters, act) =#
  n.output = n.func(act)
end

function update_cppn!(cppn::CPPN, inputs::Vector{Float64})
  for i in 1:minimum([length(cppn.inputs), length(inputs)])
    cppn.inputs[i].output = inputs[i]
  end

  for i = 1:10
    for n in cppn.hidden
      update_neuron!(n)
    end
    for n in cppn.outputs
      update_neuron!(n)
    end
  end
  map(o->o.output, cppn.outputs)
end

create_cppn()     = CPPN([], [], [])
gauss(x,x0,sigma) = exp(-(x-x0)^2 / (2.0 * sigma^2))
sigmoid(x)        = 1.0 / (1.0 + exp(-x))
id                = x->x

#= functions = { =#
#= :gauss => (p::Vector{Float64}, v::Float64)->gauss(v, p[2], p[1]), =#
#= :sin   => (p::Vector{Float64}, v::Float64)->sin(     p[1] * v + p[2]), =#
#= :cos   => (p::Vector{Float64}, v::Float64)->cos(     p[1] * v + p[2]), =#
#= :ncos  => (p::Vector{Float64}, v::Float64)->(-cos(   p[1] * v + p[2])), =#
#= :tanh  => (p::Vector{Float64}, v::Float64)->tanh(    p[1] * v + p[2]), =#
#= :sigm  => (p::Vector{Float64}, v::Float64)->sigmoid( p[1] * v + p[2]), =#
#= :id    => (p::Vector{Float64}, v::Float64)->id(      p[1] * v + p[2])} =#

#= function add_input_neuron!(cppn::CPPN,  f::Symbol, parameters::Vector{Float64}) =#
  #= n = Neuron([], [], functions[f], 0.0, parameters) =#
  #= cppn.inputs  = [cppn.inputs, n] =#
  #= cppn.neurons = [cppn.neurons, n] =#
#= end =#

#= function add_output_neuron!(cppn::CPPN, f::Symbol, parameters::Vector{Float64}) =#
  #= n = Neuron([], [], functions[f], 0.0, parameters) =#
  #= cppn.outputs = [cppn.outputs, n] =#
  #= cppn.neurons = [cppn.neurons, n] =#
#= end =#

#= function add_hidden_neuron!(cppn::CPPN, f::Symbol, parameters::Vector{Float64}) =#
  #= n = Neuron([], [], functions[f], 0.0, parameters) =#
  #= cppn.hidden  = [cppn.hidden,  n] =#
  #= cppn.neurons = [cppn.neurons, n] =#
#= end =#

functions = {
:gauss => (v::Float64)->gauss(v, 0.0, 1.0),
:sin   => (v::Float64)->  sin(v),
:nsin  => (v::Float64)->(-sin(v)),
:cos   => (v::Float64)->  cos(v),
:ncos  => (v::Float64)->(-cos(v)),
:tanh  => (v::Float64)->tanh(v),
:sigm  => (v::Float64)->sigmoid(v),
:id    => (v::Float64)->id(v)}

function add_input_neuron!(cppn::CPPN,  f::Symbol)
  n = Neuron([], [], functions[f], 0.0)
  cppn.inputs  = [cppn.inputs, n]
  cppn.neurons = [cppn.neurons, n]
end

function add_output_neuron!(cppn::CPPN, f::Symbol)
  n = Neuron([], [], functions[f], 0.0)
  cppn.outputs = [cppn.outputs, n]
  cppn.neurons = [cppn.neurons, n]
end

function add_hidden_neuron!(cppn::CPPN, f::Symbol)
  n = Neuron([], [], functions[f], 0.0)
  cppn.hidden  = [cppn.hidden,  n]
  cppn.neurons = [cppn.neurons, n]
end

function add_synapse!(cppn::CPPN, src::Neuron, dest::Neuron, weight::Float64)
  dest.inputs  = [dest.inputs,  src]
  dest.weights = [dest.weights, weight]
end

function phenotype(genome::Genome)
  cppn = CPPN([], [], [], [])
  for n in genome.neurons
    if n.ntype == :input
      #= add_input_neuron!(cppn, n.func, n.parameters) =#
      add_input_neuron!(cppn, n.func)
    elseif n.ntype == :output
      #= add_output_neuron!(cppn, n.func, n.parameters) =#
      add_output_neuron!(cppn, n.func)
    elseif n.ntype == :hidden
      #= add_hidden_neuron!(cppn, n.func, n.parameters) =#
      add_hidden_neuron!(cppn, n.func)
    else
      throw(HyperNEATError("cppn.jl: unknown neuron type given in function phenotype(...)"))
    end
  end
  for s in genome.synapses
    if s.active == true
      add_synapse!(cppn, cppn.neurons[s.src], cppn.neurons[s.dest], s.weight)
    end
  end
  cppn
end

end
