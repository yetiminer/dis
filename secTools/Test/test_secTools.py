
import yaml
import os
from secTools.secTools import SecLoader
from pytest import approx, raises

def test_import():
	with open(os.path.join(os.path.dirname(__file__),'Fixtures','samples.yml')) as fixtures_file:
		fixtures=yaml.load(fixtures_file)
		
		for fixture in fixtures:
			henry=SecLoader(fixture,'henry')
			shape=list(henry.df.shape)
			assert shape==fixture['shape']



# def test_answer():
# #test program calculations for some known examples
    # with open(os.path.join(os.path.dirname(__file__),
            # 'fixtures','samples.yaml')) as fixtures_file:
        # fixtures=yaml.load(fixtures_file)
        # for fixture in fixtures:
            # answer=fixture.pop('answer')
            # #set default values for approx function
            # relT=0.1
            # absT=0.1
            # if 'rel' in fixtures: #update values for approx function if specified
                # relT=fixtures.pop('rel')
                # absT=fixtures.pop('abs')
            # assert hunter(fixture)== approx(answer, rel=relT, abs=absT)

def test_input_data():
#test how program fails when invoked incorrectly
    with open(os.path.join(os.path.dirname(__file__),
            'fixtures','samplesErrorInput.yml')) as fixtures_file:
        fixtures=yaml.load(fixtures_file)
        for fixture in fixtures:

            with raises(AssertionError) as exception: 
                    SecLoader(fixture,'henry')

					
from secTools.secTools import yamlLoad
def test_yamlload():
#test yaml load
	pat=os.path.join(os.path.dirname(__file__),'fixtures','samplesErrorInput.yml')
	print(pat)
	yamlLoad(pat)
	
	pat=os.path.join(os.path.dirname(__file__),'fixtures','error_yaml.yml')
	with raises(UnboundLocalError) as exception: 
		yamlLoad(pat)
	