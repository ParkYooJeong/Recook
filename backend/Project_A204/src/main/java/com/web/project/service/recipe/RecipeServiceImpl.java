package com.web.project.service.recipe;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import com.web.project.dao.hashtag.HashtagDao;
import com.web.project.dao.ingredients.IngredientsSmallDao;
import com.web.project.dao.recipe.RecipeDao;
import com.web.project.dao.recipe.RecipeHashtagDao;
import com.web.project.dao.recipe.RecipeIngredientsDao;
import com.web.project.dao.survey.AllergyFoodDao;
import com.web.project.dao.survey.AllergyGroupDao;
import com.web.project.model.hashtag.Hashtag;
import com.web.project.model.ingredients.IngredientsRequest;
import com.web.project.model.ingredients.IngredientsSmall;
import com.web.project.model.recipe.Recipe;
import com.web.project.model.recipe.RecipeHashtag;
import com.web.project.model.recipe.RecipeIngredients;
import com.web.project.model.survey.AllergyFood;
import com.web.project.model.survey.AllergyGroup;

@Service
public class RecipeServiceImpl implements RecipeService {

	@Autowired
	private RecipeDao recipeDao;

	@Autowired
	private IngredientsSmallDao ingredientsSmallDao;

	@Autowired
	private RecipeIngredientsDao recipeIngredientsDao;

	@Autowired
	private HashtagDao hashtagDao;

	@Autowired
	private AllergyGroupDao allergyGroupDao;

	@Autowired
	private AllergyFoodDao allergyFoodDao;

	@Autowired
	private RecipeHashtagDao recipeHashtagDao;

	public static final Logger logger = LoggerFactory.getLogger(RecipeServiceImpl.class);

	@Override
	public void read() {

		JSONParser parser = new JSONParser();

		try {
			// JSON ?????? ??????
			String fileName = "recipe.json";

			// JSON ?????? ?????? => ??? Local PC?????? ?????? ??????
			String fileLoc = "C:\\BigData_PJT\\backend\\Project_A204\\src\\main\\resources\\json\\" + fileName;

			JSONObject jsonObject = (JSONObject) parser.parse(new FileReader(fileLoc));

			// "recipe_list" key ????????? ?????? recipe ?????? ?????????
			JSONArray recipeList = (JSONArray) jsonObject.get("recipe_list");
			for (int i = 0; i < recipeList.size(); i++) {

				// recipe ?????? ??????
				Recipe recipe = new Recipe();

				JSONObject result = (JSONObject) recipeList.get(i);

//				System.out.println("#" + i + " Recipe Read");

				Recipe skipRecipe = recipeDao.findRecipeByRecipeTitle((String) result.get("recipe_title"));

				// ?????? ????????? ?????? ???????????? ??????
				if (skipRecipe == null) {
					// ????????? ??????
					String recipeTitle = (String) result.get("recipe_title");
					recipe.setRecipeTitle(recipeTitle);
//					System.out.println("recipeTitle : " + recipeTitle);

					// ???????????? ?????? ?????? ( x?????? )
					String recipeTime = (String) result.get("recipe_time");
					recipe.setRecipeTime(recipeTime);
//					System.out.println("recipeTime : " + recipeTime);

					// ????????? ?????? ??? ?????? + ?????? ??????
					String recipeIngredientString = (String) result.get("recipe_ingredient_string");
					recipe.setRecipeIngredient(recipeIngredientString);
//					System.out.println("recipeIngredientString : " + recipeIngredientString);

					// ????????? ??????
					String recipeContext = (String) result.get("recipe_context");
					recipe.setRecipeContext(recipeContext);
//					System.out.println("recipeContext : " + recipeContext);

					// ????????? ?????????
					String recipeImage = (String) result.get("recipe_image");
					recipe.setRecipeImage(recipeImage);
//					System.out.println("recipeImage : " + recipeImage);

					// ????????? ?????? ?????????
					StringTokenizer st = new StringTokenizer(recipeImage, "####");
					String mainImage = st.nextToken();
					recipe.setRecipeMainImage(mainImage);

					// recipe DB??? ??????
//					System.out.print("recipe : ");
					Recipe resultRecipe = recipeDao.save(recipe);
//					System.out.println(resultRecipe);

					// ????????? ?????? => Recipe_Ingredients ????????????
					JSONArray recipeIngredientList = (JSONArray) result.get("recipe_ingredient");
//					System.out.print("recipeIngredientList : ");
//					System.out.println(recipeIngredientList);

					// ??? ???????????? ?????? ????????? ??????????????? ????????? ??????
					// ??????, ????????? ????????? ???????????? ()??? ????????? ???????????? ????????????
					for (int j = 0; j < recipeIngredientList.size(); j++) {
						String ingredient = (String) recipeIngredientList.get(j);

						// () ????????? ?????? => ( ???????????? ????????????
						if (ingredient.contains("(")) {
							st = new StringTokenizer(ingredient, " (");
							ingredient = st.nextToken();
						}

						IngredientsSmall small = ingredientsSmallDao.findIngredientsSmallBySmallName(ingredient);

						RecipeIngredients recipeIngredients = new RecipeIngredients();
						if (small == null) { // ???????????? ????????? ??????!
							IngredientsSmall ingredientsSmall = new IngredientsSmall();
							ingredientsSmall.setMidId(120); // 120?????? ???????????? ????????? ?????? ????????? ?????? ???
							ingredientsSmall.setSmallName(ingredient);
							small = ingredientsSmallDao.save(ingredientsSmall);
//							System.out.println(small);
						}
						recipeIngredients.setRecipeId(resultRecipe.getRecipeId());
						recipeIngredients.setSmallId(small.getSmallId());
						recipeIngredientsDao.save(recipeIngredients);
					}

					// ???????????? ????????? ???????????? -> hashtag ???????????? recipe_hashtag ????????????
					JSONArray recipeHashtagList = (JSONArray) result.get("recipe_hashtag");
//					System.out.print("recipeHashtagList : ");
//					System.out.println(recipeHashtagList);

					for (int j = 0; j < recipeHashtagList.size(); j++) {
						String recipehashtag = (String) recipeHashtagList.get(j);

						// 1. ?????? ??????????????? ???????????? ??????????????? ??????
						Hashtag hashtagOpt = hashtagDao.findHashtagByHashtagName(recipehashtag);

						Hashtag resultHashtag = null;
						if (hashtagOpt != null) {
							// 2. ????????? ???????????? count + 1
							hashtagOpt.setHashtagCount(hashtagOpt.getHashtagCount() + 1);
							resultHashtag = hashtagDao.save(hashtagOpt);
						} else {
							// 3. ????????? ???????????? ????????? ??????
							Hashtag hashtag = new Hashtag();
							hashtag.setHashtagCount(1);
							hashtag.setHashtagName(recipehashtag);
							resultHashtag = hashtagDao.save(hashtag);
						}

						// 4. ???????????? ??????????????? ??????
						RecipeHashtag recipeHashtag = new RecipeHashtag();
						recipeHashtag.setHashtagId(resultHashtag.getHashtagId());
						recipeHashtag.setRecipeId(resultRecipe.getRecipeId());
						recipeHashtagDao.save(recipeHashtag);
					}
				}
//				System.out.println("===============================");
			}
		} catch (Exception e) {
			logger.error("JSON File ???????????? ?????? : {}", e);
		}
	}

	@Override
	public ResponseEntity<Map<String, Object>> showRecipeByRecipeId(int recipeId) {
		Map<String, Object> resultMap = new HashMap<>();
		HttpStatus status = null;

		Recipe recipe = recipeDao.findRecipeByRecipeId(recipeId);

		try {
			if (recipe != null) {
				// ????????? ?????????
				resultMap.put("recipe-id", recipe.getRecipeId());
				// ????????? ??????
				resultMap.put("recipe-title", recipe.getRecipeTitle());
				// ????????? ?????? ??????
				resultMap.put("recipe-created", recipe.getRecipeCreated());
				// ????????? ?????????
				resultMap.put("recipe-image", recipe.getRecipeImage());
				// ????????? ??????
				resultMap.put("recipe-context", recipe.getRecipeContext());
				// ????????? ??????
				resultMap.put("recipe-ingredient", recipe.getRecipeIngredient());
				StringTokenizer st = new StringTokenizer(recipe.getRecipeTime(), "\n");
				// ????????? ??????
				resultMap.put("recipe-time", st.nextToken());
				// ????????? ??????
				resultMap.put("recipe-person", st.nextToken());
				// ????????? ?????? ??????
				resultMap.put("recipe-main-image", recipe.getRecipeMainImage());
				// ????????? ?????? ID
				resultMap.put("recipe-sub-id", recipe.getRecipeSubId());
				resultMap.put("recipe-count", recipe.getRecipeCount());

				status = HttpStatus.OK;
			} else {
				resultMap.put("message", "?????? ID??? ????????? ???????????? ??????");
				status = HttpStatus.INTERNAL_SERVER_ERROR;
			}
		} catch (RuntimeException e) {
			logger.error("????????? ?????? ?????? ?????? : {}", e);
			resultMap.put("message", e.getMessage());
			status = HttpStatus.INTERNAL_SERVER_ERROR;
		}

		return new ResponseEntity<Map<String, Object>>(resultMap, status);
	}

	@Override
	public ResponseEntity<Map<String, Object>> showRecipeByRecipeSubId(int recipeSubId) {
		Map<String, Object> resultMap = new HashMap<>();
		HttpStatus status = null;

		Recipe recipe = recipeDao.findRecipeByRecipeSubId(recipeSubId);

		try {
			if (recipe != null) {
				// ????????? ?????????
				resultMap.put("recipe-id", recipe.getRecipeId());
				// ????????? ??????
				resultMap.put("recipe-title", recipe.getRecipeTitle());
				// ????????? ?????? ??????
				resultMap.put("recipe-created", recipe.getRecipeCreated());
				// ????????? ?????????
				resultMap.put("recipe-image", recipe.getRecipeImage());
				// ????????? ??????
				resultMap.put("recipe-context", recipe.getRecipeContext());
				// ????????? ??????
				resultMap.put("recipe-ingredient", recipe.getRecipeIngredient());
				StringTokenizer st = new StringTokenizer(recipe.getRecipeTime(), "\n");
				// ????????? ??????
				resultMap.put("recipe-time", st.nextToken());
				// ????????? ??????
				resultMap.put("recipe-person", st.nextToken());
				// ????????? ?????? ??????
				resultMap.put("recipe-main-image", recipe.getRecipeMainImage());
				// ????????? ?????? ID
				resultMap.put("recipe-sub-id", recipe.getRecipeSubId());
				resultMap.put("recipe-count", recipe.getRecipeCount());

				status = HttpStatus.OK;
			} else {
				resultMap.put("message", "?????? ID??? ????????? ???????????? ??????");
				status = HttpStatus.INTERNAL_SERVER_ERROR;
			}
		} catch (RuntimeException e) {
			logger.error("????????? ?????? ?????? ?????? : {}", e);
			resultMap.put("message", e.getMessage());
			status = HttpStatus.INTERNAL_SERVER_ERROR;
		}

		return new ResponseEntity<Map<String, Object>>(resultMap, status);
	}

	@Override
	public ResponseEntity<List<Map<String, Object>>> newRecipeList() {
		List<Map<String, Object>> resultList = new ArrayList<Map<String, Object>>();
		HttpStatus status = HttpStatus.OK;

		List<Recipe> newList = recipeDao.findTop10ByOrderByRecipeIdDesc();

		try {
			if (newList != null) {
				for (int i = 0; i < newList.size(); i++) {
					Recipe recipe = newList.get(i);

					Map<String, Object> resultMap = new HashMap<>();

					// ????????? ?????????
					resultMap.put("recipe-id", recipe.getRecipeId());
					// ????????? ??????
					resultMap.put("recipe-title", recipe.getRecipeTitle());
					// ????????? ?????? ??????
					resultMap.put("recipe-created", recipe.getRecipeCreated());
					// ????????? ?????????
					resultMap.put("recipe-image", recipe.getRecipeImage());
					// ????????? ??????
					resultMap.put("recipe-context", recipe.getRecipeContext());
					// ????????? ??????
					resultMap.put("recipe-ingredient", recipe.getRecipeIngredient());
					StringTokenizer st = new StringTokenizer(recipe.getRecipeTime(), "\n");
					// ????????? ??????
					resultMap.put("recipe-time", st.nextToken());
					// ????????? ??????
					resultMap.put("recipe-person", st.nextToken());
					// ????????? ?????? ??????
					resultMap.put("recipe-main-image", recipe.getRecipeMainImage());

					resultList.add(resultMap);
				}
				status = HttpStatus.OK;
			} else {
				status = HttpStatus.INTERNAL_SERVER_ERROR;
			}
		} catch (RuntimeException e) {
			logger.error("????????? ?????? ?????? ?????? : {}", e);
			status = HttpStatus.INTERNAL_SERVER_ERROR;
		}

		return new ResponseEntity<List<Map<String, Object>>>(resultList, status);
	}

	@Override
	public ResponseEntity<Page<Recipe>> newRecipeListAll(Pageable pageable) {
		return new ResponseEntity<Page<Recipe>> (recipeDao.findAllByOrderByRecipeIdDesc(pageable), HttpStatus.OK);
	}
	
	@Override
	public ResponseEntity<List<Map<String, Object>>> hotRecipeList() {
		List<Map<String, Object>> resultList = new ArrayList<Map<String, Object>>();
		HttpStatus status = HttpStatus.OK;

		List<Recipe> hotList = recipeDao.findTop10ByOrderByRecipeCountDesc();

		try {
			if (hotList != null) {
				for (int i = 0; i < hotList.size(); i++) {
					Recipe recipe = hotList.get(i);

					Map<String, Object> resultMap = new HashMap<>();

					// ????????? ?????????
					resultMap.put("recipe-id", recipe.getRecipeId());
					// ????????? ??????
					resultMap.put("recipe-title", recipe.getRecipeTitle());
					// ????????? ?????? ??????
					resultMap.put("recipe-created", recipe.getRecipeCreated());
					// ????????? ?????????
					resultMap.put("recipe-image", recipe.getRecipeImage());
					// ????????? ??????
					resultMap.put("recipe-context", recipe.getRecipeContext());
					// ????????? ??????
					resultMap.put("recipe-ingredient", recipe.getRecipeIngredient());
					StringTokenizer st = new StringTokenizer(recipe.getRecipeTime(), "\n");
					// ????????? ??????
					resultMap.put("recipe-time", st.nextToken());
					// ????????? ??????
					resultMap.put("recipe-person", st.nextToken());
					// ????????? ?????? ??????
					resultMap.put("recipe-main-image", recipe.getRecipeMainImage());

					resultList.add(resultMap);
				}
				status = HttpStatus.OK;
			} else {
				status = HttpStatus.INTERNAL_SERVER_ERROR;
			}
		} catch (RuntimeException e) {
			logger.error("????????? ?????? ?????? ?????? : {}", e);
			status = HttpStatus.INTERNAL_SERVER_ERROR;
		}

		return new ResponseEntity<List<Map<String, Object>>>(resultList, status);
	}
	
	@Override
	public ResponseEntity<Page<Recipe>> hotRecipeListAll(Pageable pageable) {
		return new ResponseEntity<Page<Recipe>> (recipeDao.findAllByOrderByRecipeCountDesc(pageable), HttpStatus.OK);
	}

	@Override
	public ResponseEntity<List<String>> recipeIngredients(int recipeId) {
		List<String> resultList = new ArrayList<String>();
		HttpStatus status = null;

		List<RecipeIngredients> recipeIngredientsList = recipeIngredientsDao.findAllByRecipeId(recipeId);

		try {
			for (int i = 0; i < recipeIngredientsList.size(); i++) {
				resultList.add(ingredientsSmallDao
						.findIngredientsSmallBySmallId(recipeIngredientsList.get(i).getSmallId()).getSmallName());
			}

			status = HttpStatus.OK;
		} catch (Exception e) {
			logger.error("?????? ?????? ?????? : {}", e);
			status = HttpStatus.INTERNAL_SERVER_ERROR;
		}

		return new ResponseEntity<List<String>>(resultList, status);
	}

	@Override
	public ResponseEntity<List<Recipe>> selectIngredinets(IngredientsRequest ingredientsRequest, Pageable pageable) {
		List<Recipe> resultList = new ArrayList<Recipe>();
		HttpStatus status = null;

		int ingredientsSize = ingredientsRequest.getIngredientList().size();

		try {
			// ????????? ????????? ?????? ????????? ????????????

			Page<Recipe> recipeList = null;

			if (ingredientsSize == 1) {
				recipeList = recipeDao.findRecipeWithIngredientOne(ingredientsRequest.getIngredientList().get(0),
						pageable);
			} else if (ingredientsSize == 2) {
				recipeList = recipeDao.findRecipeWithIngredientTwo(ingredientsRequest.getIngredientList().get(0),
						ingredientsRequest.getIngredientList().get(1), pageable);
			} else if (ingredientsSize == 3) {
				recipeList = recipeDao.findRecipeWithIngredientThree(ingredientsRequest.getIngredientList().get(0),
						ingredientsRequest.getIngredientList().get(1), ingredientsRequest.getIngredientList().get(2),
						pageable);
			} else {
				return new ResponseEntity<List<Recipe>>(resultList, HttpStatus.BAD_REQUEST);
			}

			Iterator<Recipe> iterator = recipeList.iterator();
			while (iterator.hasNext()) {
				resultList.add(iterator.next());
			}

			status = HttpStatus.OK;
		} catch (Exception e) {
			logger.error("????????? ????????? ?????? ?????? : {}", e);
			status = HttpStatus.INTERNAL_SERVER_ERROR;
		}

		return new ResponseEntity<List<Recipe>>(resultList, status);
	}

	@Override
	public ResponseEntity<List<Recipe>> selectIngredinetsWithAllergy(IngredientsRequest ingredientsRequest,
			Pageable pageable) {
		List<Recipe> resultList = new ArrayList<Recipe>();
		HttpStatus status = null;

		int ingredientsSize = ingredientsRequest.getIngredientList().size();

		try {
			// ????????? ????????? ?????? ????????? ????????????

			Page<Recipe> recipeList = null;

			if (ingredientsSize == 1) {
				recipeList = recipeDao.findRecipeWithIngredientOne(ingredientsRequest.getIngredientList().get(0),
						pageable);
			} else if (ingredientsSize == 2) {
				recipeList = recipeDao.findRecipeWithIngredientTwo(ingredientsRequest.getIngredientList().get(0),
						ingredientsRequest.getIngredientList().get(1), pageable);
			} else if (ingredientsSize == 3) {
				recipeList = recipeDao.findRecipeWithIngredientThree(ingredientsRequest.getIngredientList().get(0),
						ingredientsRequest.getIngredientList().get(1), ingredientsRequest.getIngredientList().get(2),
						pageable);
			} else {
				return new ResponseEntity<List<Recipe>>(resultList, HttpStatus.BAD_REQUEST);
			}

			// ????????? ??????????????? ???????????? ???????????? ????????? ?????? ????????? ????????? ?????? ?????? ???????????????
			// 1. ID??? ???????????? ?????? ????????????
			List<AllergyGroup> allergyGroupList = allergyGroupDao.findAllByUserId(ingredientsRequest.getUserId());

			List<String> allergyIngredientList = new ArrayList<String>();

			// 2. ????????? ?????? ??????
			for (int i = 0; i < allergyGroupList.size(); i++) {
				List<AllergyFood> allergyFoodList = allergyFoodDao
						.findAllByAllergyId(allergyGroupList.get(i).getAllergyId());

				for (int j = 0; j < allergyFoodList.size(); j++) {
					allergyIngredientList.add(ingredientsSmallDao
							.findIngredientsSmallBySmallId(allergyFoodList.get(j).getSmallId()).getSmallName());
				}
			}

			Iterator<Recipe> iterator = recipeList.iterator();
			O: while (iterator.hasNext()) {
				Recipe recipe = iterator.next();
				
				for (int i = 0; i < allergyIngredientList.size(); i++) {
					if(recipe.getRecipeIngredient().contains(allergyIngredientList.get(i))) {
						continue O;
					}
				}
				
				resultList.add(recipe);
			}

			status = HttpStatus.OK;
		} catch (Exception e) {
			logger.error("????????? ????????? ?????? ?????? : {}", e);
			status = HttpStatus.INTERNAL_SERVER_ERROR;
		}

		return new ResponseEntity<List<Recipe>>(resultList, status);
	}

	@Override
	public ResponseEntity<Page<Recipe>> allRecipeByTitle(String title, Pageable pageable) {		
		return new ResponseEntity<Page<Recipe>> (recipeDao.findAllByTitle(title, pageable), HttpStatus.OK);
	}

}
