
import yaml
import os
from secTools.secTools import SecLoader
from pytest import approx, raises

def test_import():
	with open(os.path.join(os.path.dirname(__file__),'Fixtures','samples.yml')) as fixtures_file:
		fixtures=yaml.load(fixtures_file)
		henry=SecLoader(fixtures[0],'henry')
		shape=list(henry.df.shape)
		assert shape==fixtures[0]['shape']



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

# def test_input_data():
# #test how program fails when invoked incorrectly
    # with open(os.path.join(os.path.dirname(__file__),
            # 'fixtures','value_samples.yaml')) as fixtures_file:
        # fixtures=yaml.load(fixtures_file)
        # for fixture in fixtures:
            # if 'samples' in fixture: #some tests specify samples input when relevant
                # samples=fixture.pop('samples')
            # else:
                # samples=[]
            # with raises(TypeError) as exception: 
                    # hunter(fixtures,samples)

