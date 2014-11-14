module HyperNEAT

using IniFile

export CPPN

type cfg_t
  prob_synapse_change_weight::Float64 # 1 - prob_synapse_change_weight = new random value
  prob_synapse_new::Float64
  prob_neuron_new::Float64
  prob_neuron_change_function::Float64
  prob_deactivate_synapse::Float64
  prob_activate_synapse::Float64
  value_new_synapse::Float64
  delta_weight::Float64
  c1::Float64
  c2::Float64
  c3::Float64
  c4::Float64
  r::Float64 # mathing percentage
  threshold::Float64 # speciation threshold
  innovation::Int64 # global innovation number
  id::Int64
  life_time::Int64
end


function read_cfg(filename::String)
  ini = Inifile()
  read(ini, filename)
  pscw = float(get(ini, "Mutation",                      "probability synapse change weight",  -1.0))
  psn  = float(get(ini, "Mutation",                      "probability synapse new",            -1.0))
  pnn  = float(get(ini, "Mutation",                      "probability neuron new",             -1.0))
  pnc  = float(get(ini, "Mutation",                      "probability neuron change function", -1.0))
  pds  = float(get(ini, "Mutation",                      "probability deactivate synapse",     -1.0))
  pas  = float(get(ini, "Mutation",                      "probability activate synapse",       -1.0))
  vns  = float(get(ini, "Mutation",                      "value new synapse",                  -1.0))
  dw   = float(get(ini, "Mutation",                      "delta weight",                       -1.0))
  c1   = float(get(ini, "Speciation",                    "c1",                                 -1.0))
  c2   = float(get(ini, "Speciation",                    "c2",                                 -1.0))
  c3   = float(get(ini, "Speciation",                    "c3",                                 -1.0))
  c4   = float(get(ini, "Speciation",                    "c4",                                 -1.0))
  th   = float(get(ini, "Speciation",                    "threshold",                          -1.0))
  r    = float(get(ini, "Reproduction",                  "r",                                  -1.0))
  lt   = int(get(ini,   "General",                       "life time",                          -1.0))
  #ffc  = get(ini, "fitness function coefficients", "c",                                  "-1.0")
  cfg = cfg_t(
  pscw,
  psn,
  pnn,
  pnc,
  pds,
  pas,
  vns,
  dw,
  c1,
  c2,
  c3,
  c4,
  r,
  th,
  2, # innovation number
  0, # global id
  lt)

  return cfg
end

#= function Base.show(io::IO, cfg::cfg_t) =#
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

function write_cfg(filename::ASCIIString, cfg::cfg_t)
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
end


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
end

new_individual(;genomes=[], fitness=0.0, id=-1, parents=(-1,-1)) = Individual(genomes, fitness, id, parents)

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


function mutation_add_neuron!(genome::Genome, cfg::cfg_t)
  if length(genome.synapses) == 0
    return
  end
  if rand() < cfg.prob_neuron_new
    cfg.innovation = cfg.innovation + 1
    active_synapses = filter(s->s.active, genome.synapses)
    synapse         = active_synapses[ceil(rand() * length(active_synapses))]
    genome.neurons  = [genome.neurons, NeuronGene(cfg.innovation, :hidden, :sin)]
    neuron_index    = length(genome.neurons)
    src             = synapse.src
    synapse.src     = neuron_index
    genome.synapses = [genome.synapses, SynapseGene(src, neuron_index, 1.0, cfg.innovation, true)]
  end
end

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

function mutation_add_synapse!(genome::Genome, cfg::cfg_t)
  if rand() < cfg.prob_synapse_new
    cfg.innovation = cfg.innovation + 1
    l = list_of_possible_synapses(genome)
    if length(l) == 0
      return
    end
    (src,dest) = l[int(ceil(rand() * length(l)))]
    genome.synapses = [genome.synapses, SynapseGene(src, dest, cfg.value_new_synapse, cfg.innovation, true)]
  end
end

function mutation_change_synapse!(genome::Genome, cfg::cfg_t)
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

function mutation_deactivate_synapse!(genome::Genome, cfg::cfg_t)
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

function mutation_activate_synapse!(genome::Genome, cfg::cfg_t)
  if rand() < cfg.prob_activate_synapse
    if length(genome.synapses) != sum(map(s->s.active?1:0, genome.synapse))
      return # there no deactivated synapses
    end

    deactivated_synapses = filter(s->s.active == false, genome.synapse)
    deactivated_synapses[ceil(rand() * length(active_synapses))].active = true

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
    if (length(ms) > 0 && length(fs) > 0) && (length(ms) != length(fs))
      throw(HyperNEATError("crossover: Number of synapses don't match"))
    end
    if length(ms) > 0 && length(fs) > 0
      cs = [rand() < 0.5? ms[i] : fs[i] for i = 1:length(ms)]
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
      child.neurons = [child.neurons, NeuronGene(n.innovation, n.ntype, n.func)]
    end
  end
  child
end


function speciation_distance(g1::Genome, g2::Genome, cfg::cfg_t)
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

  (cfg.c1 * excess + cfg.c2 * disjoint) / N + cfg.c3 * weight_difference / matching + cfg.c4 * neuron_difference / n_matching
end

function calculate_species_sizes!(population::Population)

  fitness_values = []
  for s in population.species
    for individual in s.individuals
      fitness_values = [fitness_values, individual.fitness]
    end
  end

  minimal_fitness = minimum(fitness_values)

  for s in population.species
    for individual in s.individuals
      individual.fitness = individual.fitness + abs(minimal_fitness)
    end
  end

  average_fitness = mean(fitness_values)

  if abs(average_fitness) > 1.0
    for s in population.species # new size depends 
      ss = float(sum(map(i->i.fitness, s.individuals))) 
      av = float(average_fitness)
      println(ss, " / ", av)
      s.size = round(sum(map(i->i.fitness, s.individuals)) / average_fitness)
    end
  else
    for s in population.species
      s.size = 2.0 * length(filter(i->i.fitness > average_fitness, s.individuals))
    end
  end

  if sum(map(s->s.size, population.species)) < length(population.species)
    for s in population.species
      if s.size < 1
        s.size = 1
      end
    end
  end
end

function reproduce!(population::Population, cfg::cfg_t)
  for s in population.species
    s.individuals = sort(s.individuals, by=x->x.fitness, rev=true)
  end

  calculate_species_sizes!(population)

  individuals = []

  overall_size = sum(map(s->s.size, population.species))

  if overall_size <= length(population.species)
    for s in population.species
      s.size = 5
    end
  end

  for s in population.species
    N = 1+floor(cfg.r*length(s.individuals))
    parents = s.individuals[1:N]
    for i = 1:s.size
      mother        = parents[ceil(rand() * N)]
      father        = parents[ceil(rand() * N)]

      nr_of_genomes = length(mother.genomes)

      child         = new_individual()
      child.genomes = map(i->crossover(mother.genomes[i], father.genomes[i]), [1:nr_of_genomes])
      child.parents = (mother.id, father.id)
      child.id      = cfg.id
      cfg.id        = cfg.id + 1

      for g in child.genomes
        mutation_deactivate_synapse!(g, cfg)
        mutation_add_neuron!(g, cfg)
        mutation_change_synapse!(g, cfg)
        mutation_add_synapse!(g, cfg)
      end
      individuals = [individuals, child]
    end
  end

  population.generation = population.generation + 1

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
  println("number of individuals: $(length(individuals))")
  println("number of species:     $(length(population.species))")
  population
end

type Neuron
  inputs::Vector{Neuron}
  weights::Vector{Float64}
  func::Function
  output::Float64
end

type CPPN
  inputs::Vector{Neuron}
  hidden::Vector{Neuron}
  outputs::Vector{Neuron}
  neurons::Vector{Neuron}
end

function update_neuron!(n)
  act = sum([n.inputs[i].output * n.weights[i] for i=1:length(n.inputs)])
  n.output = convert(Float64,n.func(act))
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

create_cppn()      = CPPN([], [], [])
_gauss(x,x0,sigma) = exp(-(x-x0)^2 / (2.0 * sigma^2))
gauss(x)           = _gauss(x, 0, 0.25)
sigmoid(x)         = 1.0 / (1.0 + exp(-x))
id                 = x->x

functions = {:sin => sin, :gauss => gauss, :linear => x->x, :sigm => sigmoid, :id => id}

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
      add_input_neuron!(cppn, n.func)
    elseif n.ntype == :output
      add_output_neuron!(cppn, n.func)
    elseif n.ntype == :hidden
      add_hidden_neuron!(cppn, n. func)
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
