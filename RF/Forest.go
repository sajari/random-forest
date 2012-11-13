package RF

import (
	"log"
	"math"
	"time"
	"math/rand"
	//"fmt"
)
type Forest struct{
	Trees []*Tree
}

func BuildForest(inputs [][]interface{},labels []string, treesAmount, samplesAmount, selectedFeatureAmount int) *Forest{
	rand.Seed(time.Now().UnixNano())
	forest := &Forest{}
	forest.Trees = make([]*Tree,treesAmount)
	for i:=0;i<treesAmount;i++{
		log.Printf("building the %vth tree\n", i)
		forest.Trees[i] = BuildTree(inputs,labels,samplesAmount,selectedFeatureAmount)
	}
	log.Println("done.")
	return forest
}

func DefaultForest(inputs [][]interface{},labels []string, treesAmount int) *Forest{
	m := int( math.Sqrt( float64( len(inputs[0]) ) ) ) 
	n := int( math.Log( float64( len(inputs) ) ) / math.Log(1.3) )
	return BuildForest(inputs,labels, treesAmount,n,m)
}

func (self *Forest) Predicate(input []interface{}) string{
	counter := make(map[string]float64)
	for i:=0;i<len(self.Trees);i++{
		tree_counter := PredicateTree(self.Trees[i],input)
		total := 0.0
		for _,v := range tree_counter{
			total += float64(v)
		}
		for k,v := range tree_counter{
			counter[k] += float64(v) / total
		}
	}

	max_c := 0.0
	max_label := ""
	for k,v := range counter{
		if v>=max_c{
			max_c = v
			max_label = k
		}
	}
	return max_label
}

